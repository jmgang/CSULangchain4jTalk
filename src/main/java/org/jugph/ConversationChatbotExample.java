package org.jugph;

import dev.langchain4j.chain.ConversationalChain;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.util.Arrays;

import static dev.langchain4j.data.message.SystemMessage.systemMessage;
import static java.time.Duration.ofSeconds;

public class ConversationChatbotExample {

    public static void main(String[] args) {
        var model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .timeout(ofSeconds(120))
                .build();

        MessageWindowChatMemory memory = MessageWindowChatMemory.withMaxMessages(5);
        memory.add(systemMessage("You are a friendly conversational/instructor chatbot. " +
                "Always limit your response to 2 lines max."));

        ConversationalChain chain = ConversationalChain.builder()
                .chatLanguageModel(model)
                .chatMemory(memory)
                .build();

        var userMessages = Arrays.asList(
                "Hi, I'm Jansen. Can you give me tips to become a more efficient java developer?",
                "How many curricular programs does Cavite State University offer, and what is the total number of students and faculty across all campuses?",
                "What is the mandate of Cavite State University (CvSU) in terms of providing educational opportunities and producing individuals for global competitiveness?",
                "What's my name and what was the first tip you gave me earlier?");
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
