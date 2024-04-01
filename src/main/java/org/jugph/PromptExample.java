package org.jugph;

import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import java.util.HashMap;
import java.util.Map;

import static java.time.Duration.ofSeconds;

public class PromptExample {
    public static void main( String[] args ) {
        var model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .timeout(ofSeconds(80))
                .build();

        var template = "What are the benefits of joining a Java User Group like the Java User Group Philippines " +
                "for a {{userType}} interested in {{javaTopic}}? " +
                "Limit your answer to 2-3 lines.";
        PromptTemplate promptTemplate = PromptTemplate.from(template);

        Map<String, Object> variables = new HashMap<>();
        variables.put("userType", "experienced professionals");
        variables.put("javaTopic", "concurrency");

        Prompt prompt = promptTemplate.apply(variables);

        var response = model.generate(prompt.text());
        System.out.println(response);
    }
}
