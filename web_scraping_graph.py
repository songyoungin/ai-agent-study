"""
StateGraph 조합 방식으로 기사 검색, 추출, 요약을 수행하는 스크립트입니다.

실행 방법:
    뉴스 기사 검색 예시: uv run python web_scraping_graph.py \
        --query "인공지능 최신 동향" \
        --num_results 5 \
        --num_sentences 5
    일상 대화 예시: uv run python web_scraping_graph.py \
        --query "안녕하세요"
    그래프 시각화: uv run python web_scraping_graph.py \
        --visualize

세부 구현 사항:
    - LangGraph StateGraph로 전체 워크플로우 관리
    - 쿼리 분류 → 조건부 분기 → 뉴스 처리 (검색→추출→요약→응답) 또는 일반 응답
    - Mermaid diagram을 통한 그래프 구조 시각화 지원
"""

import argparse
from typing import Any, Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from newspaper import Article

load_dotenv(verbose=True)

# OpenAI 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)


class StateGraphDebugCallback(BaseCallbackHandler):
    """StateGraph 노드 실행을 추적하는 callback handler입니다."""

    def __init__(self):
        self.current_node = None
        self.node_count = 0

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """체인(노드) 시작 시 호출됩니다."""
        run_name = kwargs.get("name", "Unknown")

        # StateGraph 노드들만 추적
        valid_nodes = [
            "classify",
            "search_news",
            "fetch_articles",
            "summarize_articles",
            "generate_response",
            "general_response",
        ]
        if run_name in valid_nodes:
            self.current_node = run_name
            self.node_count += 1

            icons = {
                "classify": "🔍",
                "search_news": "🔎",
                "fetch_articles": "📄",
                "summarize_articles": "📝",
                "generate_response": "✨",
                "general_response": "💬",
            }

            print("=" * 50)
            print(f"{icons.get(run_name, '⚙️')} [{run_name.upper()} 노드] 실행 시작")

            # 입력 상태 출력
            if "query" in inputs:
                print(f"입력 쿼리: {inputs['query']}")
            if "is_news_related" in inputs:
                print(f"뉴스 관련성: {inputs['is_news_related']}")
            if "news_urls" in inputs and inputs["news_urls"]:
                print(f"뉴스 URL 개수: {len(inputs['news_urls'])}")
            if "articles" in inputs and inputs["articles"]:
                print(f"추출된 기사 개수: {len(inputs['articles'])}")
            if "summaries" in inputs and inputs["summaries"]:
                print(f"요약된 기사 개수: {len(inputs['summaries'])}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """체인(노드) 종료 시 호출됩니다."""
        if self.current_node:
            print(f"✅ [{self.current_node.upper()} 노드] 실행 완료")

            # 출력 상태 출력
            if "is_news_related" in outputs:
                print(f"뉴스 관련성 결과: {outputs['is_news_related']}")
            if "news_urls" in outputs and outputs["news_urls"]:
                print(f"검색된 URL 개수: {len(outputs['news_urls'])}")
            if "articles" in outputs and outputs["articles"]:
                print(f"추출된 기사 개수: {len(outputs['articles'])}")
            if "summaries" in outputs and outputs["summaries"]:
                print(f"요약된 기사 개수: {len(outputs['summaries'])}")
            if "final_response" in outputs and outputs["final_response"]:
                response_len = len(outputs["final_response"])
                print(f"응답 길이: {response_len}자")
                if response_len < 100:
                    print(f"응답 미리보기: {outputs['final_response'][:50]}...")

            print("=" * 50)
            self.current_node = None

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """체인 오류 시 호출됩니다."""
        if self.current_node:
            print(f"❌ [{self.current_node.upper()} 노드] 실행 오류: {error}")
            print("=" * 50)
            self.current_node = None


class NewsScrapingState(TypedDict):
    """
    뉴스 스크래핑 그래프의 상태를 정의합니다.
    """

    query: str  # 사용자 쿼리
    is_news_related: bool  # 뉴스 검색 필요 여부
    news_urls: List[str]  # 검색된 뉴스 URL 목록
    articles: List[str]  # 검색된 뉴스 본문 목록
    summaries: List[str]  # 요약된 뉴스 본문 목록
    final_response: str  # 최종 응답 메시지
    num_results: int  # 검색할 뉴스 기사 개수
    num_sentences: int  # 요약할 문장 개수


def classify_node(state: NewsScrapingState) -> NewsScrapingState:
    """
    사용자 쿼리가 뉴스 검색과 관련이 있는지 분류하는 노드입니다.

    Args:
        state: 현재 그래프 상태

    Returns:
        NewsScrapingState: 분류 결과가 포함된 상태
    """
    query = state["query"]

    if not query or not query.strip():
        state["is_news_related"] = False
        return state

    prompt = f"""
    다음 사용자 쿼리가 뉴스 기사 검색이 필요한 내용인지 판단해주세요.

    사용자 쿼리: "{query}"

    뉴스 검색이 필요한 경우의 예시:
    - 최신 사건, 이슈, 동향에 대한 질문
    - 특정 회사, 정치, 경제, 사회 소식
    - "뉴스", "소식", "현황", "동향" 등의 키워드 포함

    뉴스 검색이 불필요한 경우의 예시:
    - 일반적인 인사말 ("안녕", "안녕하세요")
    - 개인적인 질문이나 일상적인 대화
    - 기술적 질문이나 학습 관련 질문
    - 날씨, 음식 등 일반적인 정보 질문

    "네" 또는 "아니오"로 시작해서 한 줄로 답변해주세요.
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = str(response.content).strip().lower()

        # "네" 또는 "yes"로 시작하면 뉴스 관련, "아니오" 또는 "no"로 시작하면 일반 질문
        if answer.startswith("네") or answer.startswith("yes"):
            state["is_news_related"] = True
        elif answer.startswith("아니오") or answer.startswith("no"):
            state["is_news_related"] = False
        else:
            # 명확하지 않은 경우 기본적으로 뉴스 검색 수행
            state["is_news_related"] = True

    except Exception:
        # 오류 발생 시 안전하게 뉴스 검색 수행
        state["is_news_related"] = True

    return state


def search_news_node(state: NewsScrapingState) -> NewsScrapingState:
    """
    DuckDuckGo API를 사용해 뉴스 기사 URL을 검색하는 노드입니다.

    Args:
        state: 현재 그래프 상태

    Returns:
        NewsScrapingState: 검색된 URL 목록이 포함된 상태
    """
    query = state["query"]
    num_results = state["num_results"]

    print(
        f"검색 키워드: [{query}]에 대해 DuckDuckGo API를 이용해 기사 검색을 시작합니다. 최대 검색 결과: {num_results}개"  # noqa: E501
    )

    ddgs = DDGS()
    urls: List[str] = []

    try:
        results = ddgs.news(query, region="kr-kr", max_results=num_results)
        print(f"검색 결과: {results}")

        for item in results:
            url = item.get("url") or item.get("href") or item.get("link")
            if url and url.startswith("http"):
                urls.append(url)
                if len(urls) >= num_results:
                    break

        state["news_urls"] = urls

    except Exception as e:
        print(f"DuckDuckGo API를 이용해 기사 검색 중 오류가 발생했습니다: {e}")
        state["news_urls"] = []

    return state


def fetch_articles_node(state: NewsScrapingState) -> NewsScrapingState:
    """
    검색된 URL들에서 뉴스 본문을 추출하는 노드입니다.

    Args:
        state: 현재 그래프 상태

    Returns:
        NewsScrapingState: 추출된 기사 본문 목록이 포함된 상태
    """
    urls = state["news_urls"]
    articles: List[str] = []

    for url in urls:
        print(f"URL: {url}에 대해 뉴스 본문을 추출합니다.")

        try:
            article = Article(url)
            article.download()
            article.parse()

            if article.text and len(article.text.strip()) > 100:  # 최소 길이 확인
                articles.append(article.text)
            else:
                print(f"URL에서 충분한 본문을 추출할 수 없습니다: {url}")

        except Exception as e:
            print(f"기사 추출 중 오류 발생: {str(e)}")

    state["articles"] = articles
    return state


def summarize_articles_node(state: NewsScrapingState) -> NewsScrapingState:
    """
    추출된 기사들을 각각 요약하는 노드입니다.

    Args:
        state: 현재 그래프 상태

    Returns:
        NewsScrapingState: 요약된 기사 목록이 포함된 상태
    """
    articles = state["articles"]
    num_sentences = state["num_sentences"]
    summaries: List[str] = []

    for i, text in enumerate(articles):
        if not text or len(text.strip()) == 0:
            summaries.append("요약할 텍스트가 없습니다.")
            continue

        print(f"기사 {i + 1}/{len(articles)}: LLM을 통해 뉴스 본문을 요약합니다. 요약 문장 개수: {num_sentences}개")

        prompt = f"다음 뉴스 본문을 {num_sentences}개의 문장 이내로 한국어로 요약해 주세요:\n\n{text}\n\n요약:"

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            if isinstance(response.content, str):
                summaries.append(response.content)
            else:
                summaries.append(str(response.content))
        except Exception as e:
            summaries.append(f"요약 중 오류 발생: {str(e)}")

    state["summaries"] = summaries
    return state


def generate_response_node(state: NewsScrapingState) -> NewsScrapingState:
    """
    요약된 기사들을 종합하여 최종 응답을 생성하는 노드입니다.

    Args:
        state: 현재 그래프 상태

    Returns:
        NewsScrapingState: 최종 응답이 포함된 상태
    """
    query = state["query"]
    summaries = state["summaries"]

    if not summaries:
        state["final_response"] = f"'{query}' 관련 뉴스를 찾을 수 없었습니다."
        return state

    # 요약들을 하나의 텍스트로 결합
    combined_summaries = "\n\n".join([f"기사 {i + 1}:\n{summary}" for i, summary in enumerate(summaries)])

    prompt = f"""
    사용자가 "{query}"에 대해 질문했습니다.

    다음은 관련 뉴스 기사들을 요약한 내용입니다:

    {combined_summaries}

    이 정보들을 바탕으로 사용자의 질문에 대해 종합적이고 유용한 답변을 한국어로 작성해주세요.
    각 기사의 주요 내용을 포함하되, 자연스럽게 연결하여 하나의 완성된 답변으로 만들어주세요.
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        if isinstance(response.content, str):
            state["final_response"] = response.content
        else:
            state["final_response"] = str(response.content)
    except Exception as e:
        state["final_response"] = f"최종 응답 생성 중 오류가 발생했습니다: {str(e)}"

    return state


def general_response_node(state: NewsScrapingState) -> NewsScrapingState:
    """
    일반적인 질문에 대해 응답하는 노드입니다.

    Args:
        state: 현재 상태

    Returns:
        NewsScrapingState: 일반 응답이 포함된 상태
    """
    query = state["query"]

    prompt = f"""
    사용자가 다음과 같이 말했습니다: "{query}"

    이는 뉴스 검색이 필요하지 않은 일반적인 질문이나 대화입니다.
    친근하고 도움이 되는 방식으로 한국어로 응답해주세요.

    예시:
    - 인사말에는 인사로 답하기
    - 질문에는 도움이 되는 정보 제공
    - 대화를 이어갈 수 있는 방향으로 응답

    답변:
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        if isinstance(response.content, str):
            state["final_response"] = response.content
        else:
            state["final_response"] = str(response.content)
    except Exception as e:
        state["final_response"] = f"응답 생성 중 오류가 발생했습니다: {str(e)}"

    return state


def decide_next_step(
    state: NewsScrapingState,
) -> Literal["search_news", "general_response"]:
    """
    쿼리 분류 결과에 따라 다음 단계를 결정하는 조건부 분기 함수입니다.

    Args:
        state: 현재 상태

    Returns:
        str: 다음 노드 이름 ("search_news" 또는 "general_response")
    """
    # 라우팅 정보를 callback에서 처리하기 위해 출력
    print(f"🔀 [ROUTING] is_news_related={state['is_news_related']} → ", end="")

    if state["is_news_related"]:
        print("search_news")
        return "search_news"
    else:
        print("general_response")
        return "general_response"


def create_news_scraping_graph(num_results: int = 3, num_sentences: int = 3) -> CompiledGraph:
    """
    뉴스 스크래핑 StateGraph를 생성합니다.

    Args:
        num_results: 검색할 뉴스 기사 개수
        num_sentences: 요약할 문장 개수

    Returns:
        CompiledGraph: 컴파일된 StateGraph
    """
    # StateGraph 생성
    workflow = StateGraph(NewsScrapingState)

    # 노드 추가
    workflow.add_node("classify", classify_node)
    workflow.add_node("search_news", search_news_node)
    workflow.add_node("fetch_articles", fetch_articles_node)
    workflow.add_node("summarize_articles", summarize_articles_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("general_response", general_response_node)

    # 엣지 추가
    workflow.set_entry_point("classify")
    workflow.add_conditional_edges(
        "classify",
        decide_next_step,
        {"search_news": "search_news", "general_response": "general_response"},
    )

    # 뉴스 처리 파이프라인 연결
    workflow.add_edge("search_news", "fetch_articles")
    workflow.add_edge("fetch_articles", "summarize_articles")
    workflow.add_edge("summarize_articles", "generate_response")
    workflow.add_edge("generate_response", END)
    workflow.add_edge("general_response", END)

    # 그래프 컴파일
    return workflow.compile()


def visualize_graph(graph: CompiledGraph) -> None:
    """
    StateGraph의 구조를 다양한 방식으로 시각화합니다.

    Args:
        graph: 컴파일된 StateGraph 인스턴스
    """
    print("\n📊 [그래프 구조 시각화]")
    print("=" * 80)

    try:
        # Mermaid 다이어그램 출력
        print("\n🎨 Mermaid 다이어그램:")
        print("-" * 50)
        mermaid_code = graph.get_graph().draw_mermaid()
        print(mermaid_code)

        # ASCII 다이어그램 출력
        print("\n📝 ASCII 다이어그램:")
        print("-" * 50)
        ascii_diagram = graph.get_graph().draw_ascii()
        print(ascii_diagram)

        print("\n💡 참고사항:")
        print("- Mermaid 다이어그램은 https://mermaid.live/ 에서 시각화할 수 있습니다.")
        print("- 위 코드를 복사해서 온라인 Mermaid 에디터에 붙여넣으세요.")

    except Exception as e:
        print(f"❌ 그래프 시각화 중 오류 발생: {e}")
        print("LangGraph 버전이나 의존성을 확인해주세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StateGraph 기반 뉴스 스크래핑 에이전트")
    parser.add_argument("--query", type=str, default="인공지능 최신 동향", help="검색할 키워드")
    parser.add_argument("--num_results", type=int, default=3, help="검색할 뉴스 기사 개수")
    parser.add_argument("--num_sentences", type=int, default=3, help="요약할 문장 개수")
    parser.add_argument("--visualize", action="store_true", help="그래프 구조를 시각화하고 종료")
    args = parser.parse_args()

    # StateGraph 생성
    graph = create_news_scraping_graph()

    # 시각화 모드인 경우 그래프만 출력하고 종료
    if args.visualize:
        visualize_graph(graph)
        print("\n🎯 그래프 시각화가 완료되었습니다.")
        exit(0)

    # 초기 상태 설정
    initial_state = NewsScrapingState(
        query=args.query,
        is_news_related=False,
        news_urls=[],
        articles=[],
        summaries=[],
        final_response="",
        num_results=args.num_results,
        num_sentences=args.num_sentences,
    )

    print(f"실행 설정: 검색 키워드='{args.query}', 기사 개수={args.num_results}, 요약 문장 수={args.num_sentences}")
    print("=" * 80)

    # callback handler 생성
    debug_callback = StateGraphDebugCallback()

    # 그래프 실행 (callback 포함)
    result = graph.invoke(initial_state, config={"callbacks": [debug_callback]})

    print("\n🎯 [최종 결과]")
    print(result["final_response"])
