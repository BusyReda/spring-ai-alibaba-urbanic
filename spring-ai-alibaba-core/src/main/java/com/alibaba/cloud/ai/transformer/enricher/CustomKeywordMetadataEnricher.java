package com.alibaba.cloud.ai.transformer.enricher;

import lombok.Getter;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentTransformer;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 关键词元数据增强器 使用生成式AI模型从文档内容中提取关键词并添加到元数据中
 */
@Getter
public class CustomKeywordMetadataEnricher implements DocumentTransformer {

	/**
	 * 上下文占位符
	 */
	public static final String CONTEXT_STR_PLACEHOLDER = "context_str";

	public static final String KEYWORD_MIN_COUNT_PLACEHOLDER = "min_key_word_count";

	public static final String KEYWORD_MAX_COUNT_PLACEHOLDER = "max_key_word_count";

	/**
	 * 默认的关键词提取提示模板
	 */
	public static final String DEFAULT_KEYWORDS_TEMPLATE = """
			文档内容:{context_str}.请针对文档内容总结若干关键词，每个关键词包含2-6个字符。
			需要遵循如下规则：
			1.提取的关键词数量为{min_key_word_count}个到{max_key_word_count}个。
			2.重点关注具有业务属性的名词，主要词语语义的连贯性
			3.移除停用词（的、是、在、有、等、如何）
			4.可以进行适当的同义词扩展
			5.不要特别关注通用的，没有业务含义的词语，比如：'说明'、'定义'等
			6.关键词的语种和文档内容语种保持一致
			请用英文逗号分隔每个关键词并返回，例如：关键词1,关键词2,关键词3。""";

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
	private final int minKeywordCount;

	private final int maxKeywordCount;

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
	 * @param minKeywordCount 要提取的关键词数量下限
	 * @param maxKeywordCount 要提取的关键词数量上限
	 */
	public CustomKeywordMetadataEnricher(ChatModel chatModel, int minKeywordCount, int maxKeywordCount) {
		this(chatModel, minKeywordCount, maxKeywordCount, DEFAULT_KEYWORDS_TEMPLATE, DEFAULT_KEYWORDS_METADATA_KEY);
	}

	/**
	 * 指定元数据键名的构造函数
	 * @param chatModel AI聊天模型
	 * @param minKeywordCount 要提取的关键词数量下限
	 * @param maxKeywordCount 要提取的关键词数量上限
	 * @param keywordsMetadataKey 存储关键词的元数据键名
	 */
	public CustomKeywordMetadataEnricher(ChatModel chatModel, int minKeywordCount, int maxKeywordCount,
			String keywordsMetadataKey) {
		this(chatModel, minKeywordCount, maxKeywordCount, DEFAULT_KEYWORDS_TEMPLATE, keywordsMetadataKey);
	}

	/**
	 * 完全自定义的构造函数
	 * @param chatModel AI聊天模型
	 * @param minKeywordCount 要提取的关键词数量的下限
	 * @param maxKeywordCount 要提取的关键词数量的上限
	 * @param keywordsTemplate 关键词提取的提示模板
	 * @param keywordsMetadataKey 存储关键词的元数据键名
	 */
	public CustomKeywordMetadataEnricher(ChatModel chatModel, int minKeywordCount, int maxKeywordCount,
			String keywordsTemplate, String keywordsMetadataKey) {
		Assert.notNull(chatModel, "ChatModel must not be null");
		Assert.isTrue(maxKeywordCount >= 1, "Keyword count must be >= 1");
		Assert.isTrue(maxKeywordCount <= 20, "Keyword count must be <= 20");
		Assert.hasText(keywordsTemplate, "Keywords template must not be empty");
		Assert.hasText(keywordsMetadataKey, "Keywords metadata key must not be empty");

		this.chatModel = chatModel;
		this.minKeywordCount = minKeywordCount;
		this.maxKeywordCount = maxKeywordCount;
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
		PromptTemplate template = new PromptTemplate(this.keywordsTemplate);

		// 创建提示
		Prompt prompt = template
			.create(Map.of(CONTEXT_STR_PLACEHOLDER, document.getText(), KEYWORD_MIN_COUNT_PLACEHOLDER,
					this.minKeywordCount, KEYWORD_MAX_COUNT_PLACEHOLDER, this.maxKeywordCount));

		// 调用AI模型提取关键词
		String keywords = this.chatModel.call(prompt).getResult().getOutput().getText();

		// 将关键词添加到文档元数据中
		document.getMetadata().put(this.keywordsMetadataKey, cleanKeywords(keywords));
	}

	/**
	 * 清理关键词字符串
	 * @param keywords 原始关键词字符串
	 * @return 清理后的关键词字符串
	 */
	private List<String> cleanKeywords(String keywords) {
		if (!StringUtils.hasText(keywords)) {
			return new ArrayList<>();
		}
		String[] keywordList = keywords.split(",");
		if (keywordList.length == 0) {
			return new ArrayList<>();
		}
		return List.of(keywordList);
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

		private int maxKeywordCount = 5; // 默认提取5个关键词

		private int minKeywordCount = 3; // 默认提取3个关键词

		private String keywordsTemplate = DEFAULT_KEYWORDS_TEMPLATE;

		private String keywordsMetadataKey = DEFAULT_KEYWORDS_METADATA_KEY;

		public KeywordMetadataEnricherBuilder(ChatModel chatModel) {
			Assert.notNull(chatModel, "ChatModel must not be null");
			this.chatModel = chatModel;
		}

		/**
		 * 设置关键词数量
		 * @param minKeywordCount 关键词数量
		 * @return builder实例
		 */
		public KeywordMetadataEnricherBuilder minKeywordCount(int minKeywordCount) {
			Assert.isTrue(minKeywordCount >= 1, "Keyword count must be >= 1");
			this.minKeywordCount = minKeywordCount;
			return this;
		}

		/**
		 * 设置关键词数量
		 * @param maxKeywordCount 关键词数量
		 * @return builder实例
		 */
		public KeywordMetadataEnricherBuilder maxKeywordCount(int maxKeywordCount) {
			Assert.isTrue(maxKeywordCount <= 20, "Keyword count must be <= 20");
			this.maxKeywordCount = maxKeywordCount;
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
			return new CustomKeywordMetadataEnricher(chatModel, minKeywordCount, maxKeywordCount, keywordsTemplate,
					keywordsMetadataKey);
		}

	}

}
