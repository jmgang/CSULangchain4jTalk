package org.jugph;

import dev.langchain4j.model.input.structured.StructuredPrompt;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.output.structured.Description;
import dev.langchain4j.service.AiServices;

import java.util.Arrays;
import java.util.List;

import static java.time.Duration.ofSeconds;

public class ResourceFinderAIServiceExample {
    public record JavaBook(
            @Description("the title of the learning resource, concise") String title,
            @Description("a brief summary of the book content, up to two sentences") String summary,
            @Description("difficulty level, categorized as Beginner, Intermediate, or Advanced") String difficultyLevel,
            @Description("a list of key topics covered in the book") List<String> topicsCovered
    ) {}

    @StructuredPrompt("Find a Java book for a {{skillLevel}} developer interested in {{topics}}")
    public record JavaBookPrompt(String skillLevel, List<String> topics) {}

    interface BookFinder {
        JavaBook findBook(JavaBookPrompt prompt);
    }

    public static void main(String[] args) {

        var model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .timeout(ofSeconds(120))
                .build();

        BookFinder finder = AiServices.create(BookFinder.class, model);

        JavaBookPrompt prompt = new JavaBookPrompt("Beginner",
                Arrays.asList("object-oriented programming", "basic Java syntax"));

        JavaBook resource = finder.findBook(prompt);
        System.out.println(resource);
    }
}
