# FLIP Java 클라이언트 SDK

FLIP 채점 서버의 계약(`openapi.json`)에서 생성한 타입 있는 Java 클라이언트.
Spring 등 소비자는 DTO를 손으로 쓰지 않고 좌표 한 줄로 받아
`grade(...)`/`health()`를 타입 함수처럼 호출한다.

- 좌표: `com.flip:flip-client:<버전>` (버전 = `openapi.json`의 `info.version`)
- 배포처: GitHub Packages (Maven) — `https://maven.pkg.github.com/UNITON-3DAYS/FLIP-AI`
- 생성기: openapi-generator 7.x, `java` / `native`(JDK `HttpClient` 기반)

## 소비자(Spring) 설정

### 1. 레지스트리 인증 — PAT 필요

GitHub Packages는 **public repo여도 익명으로 받을 수 없다.** 소비자는
`read:packages` 권한의 PAT(Personal Access Token)가 필요하다. 토큰은 커밋하지
말고 `~/.gradle/gradle.properties`에 둔다:

```properties
gpr.user=<github-username>
gpr.key=<personal-access-token: read:packages>
```

### 2. build.gradle

```gradle
repositories {
    mavenCentral()
    maven {
        url = uri("https://maven.pkg.github.com/UNITON-3DAYS/FLIP-AI")
        credentials {
            username = findProperty("gpr.user") ?: System.getenv("GITHUB_ACTOR")
            password = findProperty("gpr.key")  ?: System.getenv("GITHUB_TOKEN")
        }
    }
}

dependencies {
    implementation "com.flip:flip-client:0.1.0"   // 최신 info.version으로
}
```

### 3. 호출

```java
import com.flip.client.ApiClient;
import com.flip.client.api.DefaultApi;
import com.flip.client.model.GradeRequest;
import com.flip.client.model.GradeResponse;

ApiClient client = new ApiClient();
client.updateBaseUri("http://flip:8000");   // FLIP 서버 주소

DefaultApi api = new DefaultApi(client);

GradeRequest req = new GradeRequest()
        .track(GradeRequest.TrackEnum.WORKBOOK)  // 기본값이라 생략 가능
        .name("쎈 2-1")
        .imageBase64(base64);                    // 페이지 사진 base64 (data URI 접두사 없이)

GradeResponse res = api.grade(req);
res.getResults().forEach(r ->
        System.out.println(r.getQuestionNo() + " → " + r.getVerdict()));  // O / X / HOLD
```

> 정확한 클래스·메서드명은 생성 코드 기준이다(`operationId`가 `grade`/`health`라
> 메서드는 `grade(...)`/`health()`). 응답 대기는 페이지당 수 초~십수 초이니 HTTP
> 타임아웃을 넉넉히(≈60s) 잡을 것.

## 유지보수 (FLIP 백엔드)

### 버전 규율 — 중요

패키지 버전은 `openapi.json`의 `info.version`에서 파생된다. **API 계약을 바꾸면
`api/main.py`의 `API_VERSION`을 반드시 올리고 `openapi.json`을 재생성**해야 한다.
안 올리면 GitHub Packages가 같은 버전 재배포를 거부해 publish가 실패한다(버전
누락을 잡아주는 안전장치).

### 스펙 재생성

```bash
# 서버 코드를 바꾼 뒤 openapi.json 갱신
python -c "import json; from api.main import app; \
open('openapi.json','w',encoding='utf-8').write(json.dumps(app.openapi(), ensure_ascii=False, indent=2)+'\n')"
```

`openapi.json` 또는 `client-java/**`가 바뀐 채로 main에 머지되면
`.github/workflows/publish-sdk.yml`이 자동으로 새 버전을 publish 한다. PR에서는
생성·컴파일·로컬 배포 리허설까지만 돈다(실배포 없음).

### 로컬 확인

```bash
cd client-java
gradle build                # 생성 + 컴파일 (Gradle 8.x, JDK 17 필요)
gradle publishToMavenLocal  # 실배포 없이 좌표 해소 리허설
```

Gradle wrapper는 커밋하지 않는다(바이너리). 로컬은 설치된 Gradle 8.x를, CI는
`gradle/actions/setup-gradle`이 고정 버전을 쓴다.

## 클라이언트 라이브러리 교체

`native`(JDK HttpClient) 대신 Spring 통합이 필요하면 `client-java/build.gradle`의
`library`를 바꾼다:

| library | 런타임 | 언제 |
|---|---|---|
| `native` (현재) | JDK `HttpClient` | 의존 최소, Spring 버전 비종속. 기본. |
| `restclient` | Spring `RestClient` (동기) | Spring Boot 3 표준 동기 클라이언트 선호 시. |
| `webclient` | Spring `WebClient` (리액티브) | 논블로킹 스택일 때. |

스펙과 소비 흐름은 동일하고 생성되는 호출 클래스 구현만 달라진다. Spring 팀 선호가
정해지면 한 줄로 전환한다.
