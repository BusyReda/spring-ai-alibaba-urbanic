package com.alibaba.cloud.ai.transformer.splitter;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.springframework.ai.transformer.splitter.TextSplitter;

import io.micrometer.common.util.StringUtils;
import lombok.Builder;
import lombok.Data;

public class MarkdownHeaderTextSplitter extends TextSplitter {

	// 标题匹配正则：匹配 # 开头的标题
	private static final Pattern HEADER_PATTERN = Pattern.compile("^(#{1,6})\\s+(.+)$", Pattern.MULTILINE);

	private final SplitConfig config;

	public MarkdownHeaderTextSplitter() {
		this(SplitConfig.builder().build());
	}

	public MarkdownHeaderTextSplitter(SplitConfig config) {
		this.config = config;
	}

	/**
	 * 便捷的构建方法
	 */
	public static MarkdownHeaderTextSplitter create() {
		return new MarkdownHeaderTextSplitter();
	}

	public static MarkdownHeaderTextSplitter create(int headerLevel) {
		return new MarkdownHeaderTextSplitter(SplitConfig.builder().headerLevel(headerLevel).build());
	}

	public static MarkdownHeaderTextSplitter create(int headerLevel, int maxChunkSize) {
		return new MarkdownHeaderTextSplitter(
				SplitConfig.builder().headerLevel(headerLevel).maxChunkSize(maxChunkSize).build());
	}

	/**
	 * 创建适合向量化的分割器（使用自然语言格式的标题路径）
	 */
	public static MarkdownHeaderTextSplitter createForVectorization() {
		return new MarkdownHeaderTextSplitter(
				SplitConfig.builder().headerPathMode(HeaderPathMode.NATURAL).includeHeaderPath(true).build());
	}

	/**
	 * 创建纯净内容的分割器（不包含标题路径，最适合向量化）
	 */
	public static MarkdownHeaderTextSplitter createPureContent() {
		return new MarkdownHeaderTextSplitter(SplitConfig.builder()
			.headerPathMode(HeaderPathMode.NONE)
			.includeHeaderPath(false)
			.includeContextualTitle(false)
			.build());
	}

	/**
	 * 创建带简单标题的分割器（只包含当前章节标题）
	 */
	public static MarkdownHeaderTextSplitter createWithSimpleTitle() {
		return new MarkdownHeaderTextSplitter(SplitConfig.builder()
			.headerPathMode(HeaderPathMode.NONE)
			.includeHeaderPath(false)
			.includeContextualTitle(true)
			.build());
	}

	/**
	 * 创建Markdown格式的分割器
	 */
	public static MarkdownHeaderTextSplitter createMarkdownFormat() {
		return new MarkdownHeaderTextSplitter(
				SplitConfig.builder().headerPathMode(HeaderPathMode.MARKDOWN).includeHeaderPath(true).build());
	}

	/**
	 * 核心分割方法，继承自TextSplitter
	 */
	@Override
	protected List<String> splitText(String text) {
		if (StringUtils.isBlank(text)) {
			return Collections.emptyList();
		}

		try {
			// 1. 解析标题结构
			List<HeaderInfo> headers = parseHeaders(text);

			// 2. 按配置的标题级别分割
			List<DocumentChunk> chunks = splitByHeaders(text, headers);

			// 3. 处理过长的chunk
			if (config.isAutoSubdivide()) {
				chunks = subdivideOverSizedChunks(chunks);
			}

			// 4. 过滤过小的chunk并转换为字符串列表
			return chunks.stream()
				.filter(chunk -> chunk.getContent().length() >= config.getMinChunkSize())
				.map(this::formatChunkContent)
				.collect(Collectors.toList());

		}
		catch (Exception e) {
			// 降级：按段落简单分割
			return fallbackSplit(text);
		}
	}

	/**
	 * 解析文档中的所有标题
	 */
	private List<HeaderInfo> parseHeaders(String text) {
		List<HeaderInfo> headers = new ArrayList<>();
		Matcher matcher = HEADER_PATTERN.matcher(text);

		// 用于构建标题路径的栈
		List<HeaderInfo> headerStack = new ArrayList<>();

		while (matcher.find()) {
			int level = matcher.group(1).length(); // # 的个数就是级别
			String title = matcher.group(2).trim();
			int startIndex = matcher.start();

			// 维护标题层级栈
			// 移除比当前级别低的标题
			headerStack.removeIf(h -> h.getLevel() >= level);

			// 构建标题路径
			String parentPath = headerStack.stream().map(HeaderInfo::getTitle).collect(Collectors.joining(" > "));

			String fullPath = StringUtils.isBlank(parentPath) ? title : parentPath + " > " + title;

			HeaderInfo headerInfo = HeaderInfo.builder()
				.level(level)
				.title(title)
				.startIndex(startIndex)
				.parentPath(parentPath)
				.fullPath(fullPath)
				.build();

			headers.add(headerInfo);
			headerStack.add(headerInfo);
		}

		return headers;
	}

	/**
	 * 根据标题进行分割
	 */
	private List<DocumentChunk> splitByHeaders(String text, List<HeaderInfo> headers) {
		List<DocumentChunk> chunks = new ArrayList<>();

		// 过滤出符合分割级别的标题
		List<HeaderInfo> splitHeaders = headers.stream().filter(h -> h.getLevel() <= config.getHeaderLevel()).toList();

		if (splitHeaders.isEmpty()) {
			// 没有符合条件的标题，整个文档作为一个chunk
			chunks.add(DocumentChunk.builder()
				.title("文档内容")
				.content(text.trim())
				.level(0)
				.metadata(new HashMap<>())
				.build());
			return chunks;
		}

		for (int i = 0; i < splitHeaders.size(); i++) {
			HeaderInfo currentHeader = splitHeaders.get(i);
			HeaderInfo nextHeader = (i + 1 < splitHeaders.size()) ? splitHeaders.get(i + 1) : null;

			// 确定当前chunk的内容范围
			int startIndex = currentHeader.getStartIndex();
			int endIndex = (nextHeader != null) ? nextHeader.getStartIndex() : text.length();

			String chunkContent = text.substring(startIndex, endIndex).trim();

			if (StringUtils.isNotBlank(chunkContent)) {
				DocumentChunk chunk = DocumentChunk.builder()
					.title(currentHeader.getTitle())
					.headerPath(config.isIncludeHeaderPath() ? currentHeader.getFullPath() : null)
					.content(chunkContent)
					.level(currentHeader.getLevel())
					.metadata(createMetadata(currentHeader))
					.build();

				chunks.add(chunk);
			}
		}

		return chunks;
	}

	/**
	 * 细分过长的chunk
	 */
	private List<DocumentChunk> subdivideOverSizedChunks(List<DocumentChunk> chunks) {
		List<DocumentChunk> result = new ArrayList<>();

		for (DocumentChunk chunk : chunks) {
			if (chunk.getContent().length() <= config.getMaxChunkSize()) {
				result.add(chunk);
			}
			else {
				// 尝试按更小的标题细分
				List<DocumentChunk> subChunks = subdivideBySubHeaders(chunk);
				if (subChunks.size() > 1) {
					result.addAll(subChunks);
				}
				else {
					// 按段落细分
					result.addAll(subdivideByParagraphs(chunk));
				}
			}
		}

		return result;
	}

	/**
	 * 按子标题细分
	 */
	private List<DocumentChunk> subdivideBySubHeaders(DocumentChunk chunk) {
		if (chunk.getLevel() >= 6) {
			return Collections.singletonList(chunk); // 已经是最小标题级别
		}

		// 创建更细粒度的分割器
		SplitConfig subConfig = SplitConfig.builder()
			.headerLevel(chunk.getLevel() + 1)
			.maxChunkSize(config.getMaxChunkSize())
			.minChunkSize(config.getMinChunkSize())
			.includeHeaderPath(config.isIncludeHeaderPath())
			.headerPathMode(config.getHeaderPathMode())
			.includeContextualTitle(config.isIncludeContextualTitle())
			.autoSubdivide(false) // 避免递归
			.build();

		MarkdownHeaderTextSplitter subSplitter = new MarkdownHeaderTextSplitter(subConfig);
		List<String> subTexts = subSplitter.splitText(chunk.getContent());

		if (subTexts.size() <= 1) {
			return Collections.singletonList(chunk);
		}

		return subTexts.stream()
			.map(subText -> DocumentChunk.builder()
				.title(chunk.getTitle() + " (子章节)")
				.headerPath(chunk.getHeaderPath())
				.content(subText)
				.level(chunk.getLevel() + 1)
				.metadata(chunk.getMetadata())
				.build())
			.collect(Collectors.toList());
	}

	/**
	 * 按段落细分
	 */
	private List<DocumentChunk> subdivideByParagraphs(DocumentChunk chunk) {
		List<DocumentChunk> result = new ArrayList<>();
		String[] paragraphs = chunk.getContent().split(config.getSeparator());

		StringBuilder currentContent = new StringBuilder();
		int partIndex = 1;

		for (String paragraph : paragraphs) {
			if (currentContent.length() + paragraph.length() > config.getMaxChunkSize()) {
				if (!currentContent.isEmpty()) {
					result.add(DocumentChunk.builder()
						.title(chunk.getTitle() + " (第" + partIndex + "部分)")
						.headerPath(chunk.getHeaderPath())
						.content(currentContent.toString().trim())
						.level(chunk.getLevel())
						.metadata(chunk.getMetadata())
						.build());
					currentContent = new StringBuilder();
					partIndex++;
				}
			}

			if (!currentContent.isEmpty()) {
				currentContent.append(config.getSeparator());
			}
			currentContent.append(paragraph);
		}

		if (!currentContent.isEmpty()) {
			result.add(DocumentChunk.builder()
				.title(chunk.getTitle() + " (第" + partIndex + "部分)")
				.headerPath(chunk.getHeaderPath())
				.content(currentContent.toString().trim())
				.level(chunk.getLevel())
				.metadata(chunk.getMetadata())
				.build());
		}

		return result.isEmpty() ? Collections.singletonList(chunk) : result;
	}

	/**
	 * 创建元数据
	 */
	private Map<String, Object> createMetadata(HeaderInfo header) {
		Map<String, Object> metadata = new HashMap<>();
		metadata.put("title", header.getTitle());
		metadata.put("level", header.getLevel());
		metadata.put("header_path", header.getFullPath());
		metadata.put("parent_path", header.getParentPath());
		metadata.put("splitter", "MarkdownHeaderTextSplitter");
		return metadata;
	}

	/**
	 * 降级分割策略
	 */
	private List<String> fallbackSplit(String text) {
		String[] paragraphs = text.split(config.getSeparator());
		List<String> result = new ArrayList<>();
		StringBuilder currentChunk = new StringBuilder();

		for (String paragraph : paragraphs) {
			if (currentChunk.length() + paragraph.length() > config.getMaxChunkSize()) {
				if (!currentChunk.isEmpty()) {
					result.add(currentChunk.toString().trim());
					currentChunk = new StringBuilder();
				}
			}

			if (!currentChunk.isEmpty()) {
				currentChunk.append(config.getSeparator());
			}
			currentChunk.append(paragraph);
		}

		if (!currentChunk.isEmpty()) {
			result.add(currentChunk.toString().trim());
		}

		return result.stream().filter(chunk -> chunk.length() >= config.getMinChunkSize()).collect(Collectors.toList());
	}

	/**
	 * 分割配置
	 */
	@Data
	@Builder
	public static class SplitConfig {

		@Builder.Default
		private int headerLevel = 2; // 分割的标题级别，默认按H2分割

		@Builder.Default
		private int maxChunkSize = 2000; // 最大chunk大小（字符数）

		@Builder.Default
		private int minChunkSize = 100; // 最小chunk大小（字符数）

		@Builder.Default
		private boolean includeHeaderPath = true; // 是否包含标题路径作为上下文

		@Builder.Default
		private boolean autoSubdivide = true; // 是否自动细分过长的chunk

		@Builder.Default
		private String separator = "\n\n"; // 段落分隔符

		@Builder.Default
		private HeaderPathMode headerPathMode = HeaderPathMode.NATURAL; // 标题路径包含模式

		@Builder.Default
		private boolean includeContextualTitle = true; // 是否包含上下文标题

	}

	/**
	 * 标题路径包含模式
	 */
	public enum HeaderPathMode {

		NONE, // 不包含标题路径
		NATURAL, // 自然语言形式（推荐用于向量化）
		STRUCTURED, // 结构化形式（标题路径: XXX）
		MARKDOWN // Markdown格式

	}

	/**
	 * 标题信息
	 */
	@Data
	@Builder
	public static class HeaderInfo {

		private int level; // 标题级别

		private String title; // 标题内容

		private int startIndex; // 在原文中的起始位置

		private String parentPath; // 父级标题路径

		private String fullPath; // 完整标题路径

	}

	/**
	 * 文档块
	 */
	@Data
	@Builder
	public static class DocumentChunk {

		private String title; // 主标题

		private String headerPath; // 标题路径

		private String content; // 内容

		private int level; // 标题级别

		private Map<String, Object> metadata; // 元数据

	}

	/**
	 * 分割文本并返回DocumentChunk对象列表（提供更多控制选项）
	 */
	public List<DocumentChunk> splitToChunks(String text) {
		if (StringUtils.isBlank(text)) {
			return Collections.emptyList();
		}

		try {
			// 1. 解析标题结构
			List<HeaderInfo> headers = parseHeaders(text);

			// 2. 按配置的标题级别分割
			List<DocumentChunk> chunks = splitByHeaders(text, headers);

			// 3. 处理过长的chunk
			if (config.isAutoSubdivide()) {
				chunks = subdivideOverSizedChunks(chunks);
			}

			// 4. 过滤过小的chunk
			return chunks.stream()
				.filter(chunk -> chunk.getContent().length() >= config.getMinChunkSize())
				.collect(Collectors.toList());

		}
		catch (Exception e) {
			// 降级：返回单个chunk
			return Collections.singletonList(DocumentChunk.builder()
				.title("文档内容")
				.content(text.trim())
				.level(0)
				.metadata(new HashMap<>())
				.build());
		}
	}

	/**
	 * 获取纯净内容列表（最适合向量化）
	 */
	public List<String> splitToPureContent(String text) {
		return splitToChunks(text).stream().map(DocumentChunk::getContent).collect(Collectors.toList());
	}

	/**
	 * 获取带简单标题的内容列表
	 */
	public List<String> splitToContentWithTitle(String text) {
		return splitToChunks(text).stream().map(this::formatChunkWithTitle).collect(Collectors.toList());
	}

	/**
	 * 根据配置格式化chunk内容
	 */
	private String formatChunkContent(DocumentChunk chunk) {
		StringBuilder sb = new StringBuilder();

		// 根据配置决定如何处理标题路径
		if (config.isIncludeHeaderPath() && StringUtils.isNotBlank(chunk.getHeaderPath())) {
			switch (config.getHeaderPathMode()) {
				case NATURAL:
					// 自然语言形式，适合向量化
					sb.append("在").append(chunk.getHeaderPath().replace(" > ", "的")).append("部分：\n\n");
					break;
				case STRUCTURED:
					// 传统结构化形式
					sb.append("标题路径: ").append(chunk.getHeaderPath()).append("\n\n");
					break;
				case MARKDOWN:
					// Markdown标题形式
					String[] pathParts = chunk.getHeaderPath().split(" > ");
					for (int i = 0; i < pathParts.length; i++) {
						sb.append("#".repeat(i + 1)).append(" ").append(pathParts[i]).append("\n");
					}
					sb.append("\n");
					break;
				case NONE:
				default:
					// 不包含标题路径
					break;

			}
		}
		else if (config.isIncludeContextualTitle() && StringUtils.isNotBlank(chunk.getTitle())) {
			// 如果不包含路径但包含上下文标题，添加简单的标题上下文
			sb.append("关于").append(chunk.getTitle()).append("：\n\n");
		}

		sb.append(chunk.getContent());
		return sb.toString().trim();
	}

	/**
	 * 格式化带标题的chunk内容
	 */
	private String formatChunkWithTitle(DocumentChunk chunk) {
		if (StringUtils.isNotBlank(chunk.getTitle())) {
			return chunk.getTitle() + "\n\n" + chunk.getContent().trim();
		}
		return chunk.getContent().trim();
	}

}
