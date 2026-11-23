import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A small Java 21 client for the LibAssist REST API, with no dependencies (java.net.http only).
 *
 * <p>It shows that the API is a plain HTTP contract that any language can use, and demonstrates the
 * client side of the API's design: authentication header, Problem Details errors, Idempotency-Key with
 * safe retries, and ETag / If-Match for optimistic concurrency.
 *
 * <p>Run (single-file source launch, no build step):
 * <pre>
 *   libassist-api &amp;                                   # or: make api
 *   java clients/java/LibraryClient.java http://127.0.0.1:8000
 * </pre>
 *
 * <p>A production client would be generated from /openapi.json (openapi-generator) and parse JSON with
 * Jackson; this file keeps JSON handling minimal on purpose.
 */
public final class LibraryClient {

    /** An HTTP response: status, body and the headers we care about. */
    public record ApiResponse(int status, String body, HttpResponse<String> raw) {
        Optional<String> header(String name) {
            return raw.headers().firstValue(name);
        }
    }

    /** An error response, parsed from RFC 9457 Problem Details. */
    public static final class ApiException extends RuntimeException {
        final int status;
        final String code;

        ApiException(int status, String code, String detail) {
            super(status + " " + code + ": " + detail);
            this.status = status;
            this.code = code;
        }
    }

    private static final Set<Integer> RETRYABLE = Set.of(429, 502, 503, 504);
    private static final int MAX_ATTEMPTS = 3;

    private final HttpClient http;
    private final String baseUrl;
    private final String memberId;

    public LibraryClient(String baseUrl, String memberId) {
        this.http = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1) // plain-HTTP dev server: skip the h2c upgrade attempt
                .connectTimeout(Duration.ofSeconds(5))
                .build();
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.memberId = memberId;
    }

    // ------------------------------------------------------------------ API operations

    public ApiResponse searchPolicies(String question) {
        String q = URLEncoder.encode(question, StandardCharsets.UTF_8);
        return send(request("/v1/policies/search?q=" + q).GET(), true);
    }

    public ApiResponse account() {
        return send(request("/v1/members/" + memberId + "/account").GET(), true);
    }

    /** POST is not idempotent, so the retry is only safe because every attempt carries the same key. */
    public ApiResponse renewLoan(String loanId, String idempotencyKey) {
        HttpRequest.Builder builder = request("/v1/loans/" + loanId + "/renewals")
                .header("Idempotency-Key", idempotencyKey)
                .POST(HttpRequest.BodyPublishers.noBody());
        return send(builder, true);
    }

    public ApiResponse member() {
        return send(request("/v1/members/" + memberId).GET(), true);
    }

    /** PATCH with If-Match: the server refuses (412) if someone changed the record since we read it. */
    public ApiResponse updateEmail(String email, String etag) {
        String json = "{\"email\": \"" + email.replace("\"", "") + "\"}";
        HttpRequest.Builder builder = request("/v1/members/" + memberId)
                .header("Content-Type", "application/json")
                .header("If-Match", etag)
                .method("PATCH", HttpRequest.BodyPublishers.ofString(json));
        return send(builder, false);
    }

    // ------------------------------------------------------------------ plumbing

    private HttpRequest.Builder request(String path) {
        return HttpRequest.newBuilder(URI.create(baseUrl + path))
                .timeout(Duration.ofSeconds(10))
                .header("Accept", "application/json")
                .header("X-Member-Id", memberId); // stands in for "Authorization: Bearer <token>"
    }

    private ApiResponse send(HttpRequest.Builder builder, boolean safeToRetry) {
        HttpRequest request = builder.build();
        int attempts = safeToRetry ? MAX_ATTEMPTS : 1;
        for (int attempt = 1; ; attempt++) {
            try {
                HttpResponse<String> response = http.send(request, HttpResponse.BodyHandlers.ofString());
                int status = response.statusCode();
                if (RETRYABLE.contains(status) && attempt < attempts) {
                    backoff(attempt);
                    continue;
                }
                if (status >= 400) {
                    throw new ApiException(status, jsonString(response.body(), "code").orElse("http-" + status),
                            jsonString(response.body(), "detail").orElse(response.body()));
                }
                return new ApiResponse(status, response.body(), response);
            } catch (IOException e) {
                if (attempt >= attempts) {
                    throw new ApiException(503, "api-unreachable", e.getMessage());
                }
                backoff(attempt);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new ApiException(503, "interrupted", "request interrupted");
            }
        }
    }

    private static void backoff(int attempt) {
        try {
            Thread.sleep(200L * (1L << (attempt - 1))); // 200 ms, 400 ms, ...
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Extracts the first "key": "value" string from a JSON document. Enough for this demo; use Jackson. */
    static Optional<String> jsonString(String json, String key) {
        Matcher m = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(json);
        return m.find() ? Optional.of(m.group(1)) : Optional.empty();
    }

    // ------------------------------------------------------------------ demo

    public static void main(String[] args) {
        String baseUrl = args.length > 0 ? args[0] : "http://127.0.0.1:8000";
        LibraryClient lea = new LibraryClient(baseUrl, "M-1001");

        ApiResponse policies = lea.searchPolicies("How many times can I renew a book?");
        System.out.println("1. Top policy passage: " + jsonString(policies.body(), "citation").orElse("?"));

        ApiResponse account = lea.account();
        System.out.println("2. Fines owed: " + jsonString(account.body(), "fines_total").orElse("?"));

        String key = UUID.randomUUID().toString();
        try {
            ApiResponse renewed = lea.renewLoan("L-5001", key);
            ApiResponse retried = lea.renewLoan("L-5001", key); // e.g. after a timeout: same key, no double renewal
            System.out.println("3. Renewed L-5001, new due date " + jsonString(renewed.body(), "due_on").orElse("?")
                    + "; retry replayed=" + retried.header("Idempotent-Replayed").orElse("false")
                    + ", same body=" + renewed.body().equals(retried.body()));
        } catch (ApiException e) { // happens if you run the demo again on the same database
            System.out.println("3. Renewal of L-5001 refused: HTTP " + e.status + " " + e.code);
        }

        try {
            lea.renewLoan("L-5002", UUID.randomUUID().toString());
        } catch (ApiException e) {
            System.out.println("4. Renewal of L-5002 refused: HTTP " + e.status + " " + e.code);
        }

        String etag = lea.member().header("ETag").orElseThrow();
        ApiResponse updated = lea.updateEmail("lea.martin@uni.example.edu", etag);
        System.out.println("5. Email updated with If-Match " + etag + ", new ETag "
                + updated.header("ETag").orElse("?"));
        try {
            lea.updateEmail("someone.else@example.edu", etag); // stale ETag: a lost update is prevented
        } catch (ApiException e) {
            System.out.println("6. Second update with the stale ETag: HTTP " + e.status + " " + e.code);
        }
    }
}
