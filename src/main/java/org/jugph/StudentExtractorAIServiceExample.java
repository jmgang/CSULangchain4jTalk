package org.jugph;

import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.UserMessage;

import java.time.LocalDate;

import static java.time.Duration.ofSeconds;

public class StudentExtractorAIServiceExample {
    public record Student(String firstName, String lastName, String studentEmail, String studentID,
                          String level, String major, LocalDate registrationDate) {

        @Override
        public String toString() {
            return "Student {" +
                    " firstName = \"" + firstName + "\"" +
                    ", lastName = \"" + lastName + "\"" +
                    ", studentEmail = \"" + studentEmail + "\"" +
                    ", studentID = \"" + studentID + "\"" +
                    ", level = \"" + level + "\"" +
                    ", major = \"" + major + "\"" +
                    ", registrationDate = " + registrationDate +
                    " }";
        }
    }

    interface StudentExtractor {
        @UserMessage("Extract student information from the following text: {{it}}")
        Student extractStudentFrom(String text);
    }

    public static void main(String[] args) {

        var model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .timeout(ofSeconds(120))
                .logRequests(true)
                .logResponses(true)
                .build();

        StudentExtractor extractor = AiServices.create(StudentExtractor.class, model);

        var text = "New student alert: Juan Dela Cruz, a 2nd year Computer Engineering major, has just enrolled. " +
                "His student email, juan.delacruz@university.edu, and student ID, 123456, were registered on the 1st of September, 2023.";

        Student student = extractor.extractStudentFrom(text);

        System.out.println(student.toString());
    }
}
