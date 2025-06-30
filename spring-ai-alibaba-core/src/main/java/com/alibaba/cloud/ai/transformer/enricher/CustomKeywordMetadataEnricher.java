package com.alibaba.cloud.ai.transformer.enricher;

import java.util.List;
import java.util.Map;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentTransformer;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

import lombok.Getter;

/**
 * 关键词元数据增强器 使用生成式AI模型从文档内容中提取关键词并添加到元数据中
 */
@Getter
public class CustomKeywordMetadataEnricher implements DocumentTransformer {

	/**
	 * 上下文占位符
	 */
	public static final String CONTEXT_STR_PLACEHOLDER = "context_str";

	/**
	 * 默认的关键词提取提示模板
	 */
	public static final String DEFAULT_KEYWORDS_TEMPLATE = """
			{context_str}. Give %s unique keywords for this
			document. Format as comma separated. Keywords: """;

	/**
	 * 默认的关键词元数据键名
	 */
	public static final String DEFAULT_KEYWORDS_METADATA_KEY = "excerpt_keywords";

	/**
	 * AI聊天模型
	 */
	private final ChatModel chatModel;

	/**
	 * 要提取的关键词数量
	 */
	private final int keywordCount;

	/**
	 * 关键词提取的提示模板
	 */
	private final String keywordsTemplate;

	/**
	 * 存储关键词的元数据键名
	 */
	private final String keywordsMetadataKey;

	/**
	 * 使用默认配置的构造函数
	 * @param chatModel AI聊天模型
	 * @param keywordCount 要提取的关键词数量
	 */
	public CustomKeywordMetadataEnricher(ChatModel chatModel, int keywordCount) {
		this(chatModel, keywordCount, DEFAULT_KEYWORDS_TEMPLATE, DEFAULT_KEYWORDS_METADATA_KEY);
	}

	/**
	 * 指定元数据键名的构造函数
	 * @param chatModel AI聊天模型
	 * @param keywordCount 要提取的关键词数量
	 * @param keywordsMetadataKey 存储关键词的元数据键名
	 */
	public CustomKeywordMetadataEnricher(ChatModel chatModel, int keywordCount, String keywordsMetadataKey) {
		this(chatModel, keywordCount, DEFAULT_KEYWORDS_TEMPLATE, keywordsMetadataKey);
	}

	/**
	 * 完全自定义的构造函数
	 * @param chatModel AI聊天模型
	 * @param keywordCount 要提取的关键词数量
	 * @param keywordsTemplate 关键词提取的提示模板
	 * @param keywordsMetadataKey 存储关键词的元数据键名
	 */
	public CustomKeywordMetadataEnricher(ChatModel chatModel, int keywordCount, String keywordsTemplate,
			String keywordsMetadataKey) {
		Assert.notNull(chatModel, "ChatModel must not be null");
		Assert.isTrue(keywordCount >= 1, "Keyword count must be >= 1");
		Assert.hasText(keywordsTemplate, "Keywords template must not be empty");
		Assert.hasText(keywordsMetadataKey, "Keywords metadata key must not be empty");

		this.chatModel = chatModel;
		this.keywordCount = keywordCount;
		this.keywordsTemplate = keywordsTemplate;
		this.keywordsMetadataKey = keywordsMetadataKey;
	}

	/**
	 * 对文档列表进行关键词提取增强
	 * @param documents 待处理的文档列表
	 * @return 增强后的文档列表
	 */
	@Override
	public List<Document> apply(List<Document> documents) {
		for (Document document : documents) {
			enhanceDocumentWithKeywords(document);
		}
		return documents;
	}

	/**
	 * 为单个文档提取关键词并添加到元数据中
	 * @param document 待处理的文档
	 */
	private void enhanceDocumentWithKeywords(Document document) {
		// 检查文档内容是否为空
		if (!StringUtils.hasText(document.getText())) {
			return;
		}

		// 创建提示模板
		String formattedTemplate = String.format(this.keywordsTemplate, this.keywordCount);
		PromptTemplate template = new PromptTemplate(formattedTemplate);

		// 创建提示
		Prompt prompt = template.create(Map.of(CONTEXT_STR_PLACEHOLDER, document.getText()));

		// 调用AI模型提取关键词
		String keywords = this.chatModel.call(prompt).getResult().getOutput().getText();

		// 清理关键词字符串（去除多余的空格和换行）
		String cleanedKeywords = cleanKeywords(keywords);

		// 将关键词添加到文档元数据中
		document.getMetadata().put(this.keywordsMetadataKey, cleanedKeywords);
	}

	/**
	 * 清理关键词字符串
	 * @param keywords 原始关键词字符串
	 * @return 清理后的关键词字符串
	 */
	private String cleanKeywords(String keywords) {
		if (!StringUtils.hasText(keywords)) {
			return "";
		}

		// 去除前后空格和换行符
		return keywords.trim().replaceAll("\\s+", " ");
	}

	/**
	 * 便捷的构建器方法
	 * @param chatModel AI聊天模型
	 * @return KeywordMetadataEnricherBuilder实例
	 */
	public static KeywordMetadataEnricherBuilder builder(ChatModel chatModel) {
		return new KeywordMetadataEnricherBuilder(chatModel);
	}

	/**
	 * KeywordMetadataEnricher的构建器类
	 */
	public static class KeywordMetadataEnricherBuilder {

		private final ChatModel chatModel;

		private int keywordCount = 5; // 默认提取5个关键词

		private String keywordsTemplate = DEFAULT_KEYWORDS_TEMPLATE;

		private String keywordsMetadataKey = DEFAULT_KEYWORDS_METADATA_KEY;

		public KeywordMetadataEnricherBuilder(ChatModel chatModel) {
			Assert.notNull(chatModel, "ChatModel must not be null");
			this.chatModel = chatModel;
		}

		/**
		 * 设置关键词数量
		 * @param keywordCount 关键词数量
		 * @return builder实例
		 */
		public KeywordMetadataEnricherBuilder keywordCount(int keywordCount) {
			Assert.isTrue(keywordCount >= 1, "Keyword count must be >= 1");
			this.keywordCount = keywordCount;
			return this;
		}

		/**
		 * 设置关键词提取模板
		 * @param keywordsTemplate 关键词提取模板
		 * @return builder实例
		 */
		public KeywordMetadataEnricherBuilder keywordsTemplate(String keywordsTemplate) {
			Assert.hasText(keywordsTemplate, "Keywords template must not be empty");
			this.keywordsTemplate = keywordsTemplate;
			return this;
		}

		/**
		 * 设置关键词元数据键名
		 * @param keywordsMetadataKey 关键词元数据键名
		 * @return builder实例
		 */
		public KeywordMetadataEnricherBuilder keywordsMetadataKey(String keywordsMetadataKey) {
			Assert.hasText(keywordsMetadataKey, "Keywords metadata key must not be empty");
			this.keywordsMetadataKey = keywordsMetadataKey;
			return this;
		}

		/**
		 * 构建KeywordMetadataEnricher实例
		 * @return KeywordMetadataEnricher实例
		 */
		public CustomKeywordMetadataEnricher build() {
			return new CustomKeywordMetadataEnricher(chatModel, keywordCount, keywordsTemplate, keywordsMetadataKey);
		}

	}

}