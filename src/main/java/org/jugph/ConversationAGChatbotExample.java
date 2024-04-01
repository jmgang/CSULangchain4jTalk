package org.jugph;

import dev.langchain4j.chain.ConversationalChain;
import dev.langchain4j.data.document.loader.UrlDocumentLoader;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.transformer.HtmlTextExtractor;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.util.Arrays;
import java.util.Collections;

import static dev.langchain4j.data.message.SystemMessage.systemMessage;
import static java.time.Duration.ofSeconds;

public class ConversationAGChatbotExample {

    public static void main(String[] args) {

        var document = UrlDocumentLoader
                .load("https://cvsu.edu.ph/history/", new TextDocumentParser());

        var textExtractor = new HtmlTextExtractor();
        var transformedDocument = textExtractor.transform(document);

        //System.out.println(transformedDocument);

        Prompt systemMessagePrompt = PromptTemplate.from("Answer the following questions to the best of your ability.\n\n" +
                "You can base your answer on the following information:\n{{information}}. Limit your response to 2 lines max.")
                .apply(Collections.singletonMap("information", transformedDocument));

        var model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .timeout(ofSeconds(120))
                .build();

        MessageWindowChatMemory memory = MessageWindowChatMemory.withMaxMessages(5);
        memory.add(systemMessage(systemMessagePrompt.text()));

        ConversationalChain chain = ConversationalChain.builder()
                .chatLanguageModel(model)
                .chatMemory(memory)
                .build();

        var userMessages = Arrays.asList(
                "What was the original name of Cavite State University when it was founded in 1906, and who were the first teachers?",
                "How many curricular programs does Cavite State University offer, and what is the total number of students and faculty across all campuses?",
                "What is the mandate of Cavite State University (CvSU) in terms of providing educational opportunities and producing individuals for global competitiveness?",
                "Can you name some of the key research centers and consortia hosted by CvSU, as mentioned in their institutional profile?"
        );
        var output = new StringBuilder();

        System.out.println("============================================");
        for (String userMessage : userMessages) {
            output.append("\n[USER]: ")
                    .append(userMessage)
                    .append("\n[LLM]: ")
                    .append(chain.execute(userMessage))
                    .append("\n");
        }

        System.out.println(output);
    }
}
