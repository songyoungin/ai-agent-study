"""
Multi-Agent 구조로 기사 검색, 추출, 요약을 수행하는 스크립트입니다.

실행 방법:
    뉴스 기사 검색 예시: uv run python web_scraping_multiagent.py --query "인공지능 최신 동향" --num_results 5 --num_sentences 5
    일상 대화 예시: uv run python web_scraping_multiagent.py --query "안녕하세요"
    그래프 시각화: uv run python web_scraping_multiagent.py --visualize

세부 구현 사항:
    - LangGraph StateGraph + Multi-Agent 시스템
    - Supervisor Agent가 전체 워크플로우 조정
    - 각 Agent는 전문화된 역할 담당 (분류, 검색, 스크래핑, 요약, 응답생성)
    - Agent간 상태 공유를 통한 협업
    - Mermaid diagram을 통한 그래프 구조 시각화 지원
"""

import argparse
from typing import List, TypedDict, Any, Dict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from newspaper import Article
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
from dotenv import load_dotenv
import operator

load_dotenv(verbose=True)

# OpenAI 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)


class MultiAgentDebugCallback(BaseCallbackHandler):
    """Multi-Agent StateGraph 노드 실행을 추적하는 callback handler입니다."""

    def __init__(self):
        self.current_node = None
        self.node_count = 0

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """체인(노드) 시작 시 호출됩니다."""
        run_name = kwargs.get("name", "Unknown")

        # Multi-Agent 노드들만 추적
        valid_nodes = [
            "supervisor",
            "classifier_agent",
            "search_agent",
            "scraper_agent",
            "summarizer_agent",
            "response_generator_agent",
            "general_chat_agent",
        ]
        if run_name in valid_nodes:
            self.current_node = run_name
            self.node_count += 1

            icons = {
                "supervisor": "👨‍💼",
                "classifier_agent": "🔍",
                "search_agent": "🔎",
                "scraper_agent": "📄",
                "summarizer_agent": "📝",
                "response_generator_agent": "✨",
                "general_chat_agent": "💬",
            }

            print("=" * 60)
            print(f"{icons.get(run_name, '⚙️')} [{run_name.upper()}] 실행 시작")

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
            print(f"✅ [{self.current_node.upper()}] 실행 완료")

            # 출력 상태 출력
            if "next" in outputs:
                print(f"다음 Agent: {outputs['next']}")
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

            print("=" * 60)
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
            print(f"❌ [{self.current_node.upper()}] 실행 오류: {error}")
            print("=" * 60)
            self.current_node = None


class AgentState(TypedDict):
    """
    Multi-Agent 시스템의 공유 상태를 정의합니다.
    """

    query: str  # 사용자 쿼리
    messages: Annotated[List[BaseMessage], operator.add]  # Agent간 메시지 교환
    next: str  # 다음에 실행할 Agent 이름
    is_news_related: bool  # 뉴스 검색 필요 여부
    news_urls: List[str]  # 검색된 뉴스 URL 목록
    articles: List[str]  # 검색된 뉴스 본문 목록
    summaries: List[str]  # 요약된 뉴스 본문 목록
    final_response: str  # 최종 응답 메시지
    num_results: int  # 검색할 뉴스 기사 개수
    num_sentences: int  # 요약할 문장 개수


# =============================================================================
# 노드 함수들
# =============================================================================


def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor Agent가 전체 워크플로우를 조정합니다."""

    print(
        f"🧭 Supervisor 상태 체크: next='{state['next']}', is_news_related={state.get('is_news_related', 'None')}"
    )

    # 초기 상태: 분류 Agent로 시작 (아직 아무것도 시작하지 않았다면)
    if state["next"] == "":
        state["next"] = "classifier_agent"
        state["messages"].append(
            AIMessage(content="🏁 워크플로우 시작: 쿼리 분류부터 시작합니다.")
        )
        return state

    # 분류 완료 후 분기
    if state["next"] == "classified":
        if state["is_news_related"]:
            state["next"] = "search_agent"
            state["messages"].append(
                AIMessage(content="📈 뉴스 관련 쿼리 감지: 뉴스 검색을 시작합니다.")
            )
        else:
            state["next"] = "general_chat_agent"
            state["messages"].append(
                AIMessage(content="💭 일반 대화 감지: 일반 응답을 생성합니다.")
            )
        return state

    # 뉴스 검색 완료 후
    if state["next"] == "searched":
        state["next"] = "scraper_agent"
        state["messages"].append(
            AIMessage(content="🔗 URL 검색 완료: 기사 본문 추출을 시작합니다.")
        )
        return state

    # 스크래핑 완료 후
    if state["next"] == "scraped":
        state["next"] = "summarizer_agent"
        state["messages"].append(
            AIMessage(content="📰 본문 추출 완료: 기사 요약을 시작합니다.")
        )
        return state

    # 요약 완료 후
    if state["next"] == "summarized":
        state["next"] = "response_generator_agent"
        state["messages"].append(
            AIMessage(content="📋 요약 완료: 최종 응답을 생성합니다.")
        )
        return state

    # 모든 작업 완료
    if state["next"] == "completed":
        state["next"] = "FINISH"
        state["messages"].append(AIMessage(content="🎯 모든 작업이 완료되었습니다!"))
        return state

    # 기본값: 종료
    state["next"] = "FINISH"
    return state


def classifier_agent_node(state: AgentState) -> AgentState:
    """Classifier Agent 노드"""
    query = state["query"]

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

    state["messages"].append(
        AIMessage(
            content=f"분류 결과: {'뉴스 관련' if state['is_news_related'] else '일반 대화'}"
        )
    )
    # 분류 완료 표시
    state["next"] = "classified"
    return state


def search_agent_node(state: AgentState) -> AgentState:
    """Search Agent 노드"""
    query = state["query"]
    num_results = state["num_results"]

    print(
        f"검색 키워드: [{query}]에 대해 DuckDuckGo API를 이용해 기사 검색을 시작합니다. 최대 검색 결과: {num_results}개"
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

    state["messages"].append(AIMessage(content=f"검색 완료: {len(urls)}개 URL 발견"))
    # 검색 완료 표시
    state["next"] = "searched"
    return state


def scraper_agent_node(state: AgentState) -> AgentState:
    """Scraper Agent 노드"""
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
    state["messages"].append(
        AIMessage(content=f"스크래핑 완료: {len(articles)}개 기사 추출")
    )
    # 스크래핑 완료 표시
    state["next"] = "scraped"
    return state


def summarizer_agent_node(state: AgentState) -> AgentState:
    """Summarizer Agent 노드"""
    articles = state["articles"]
    num_sentences = state["num_sentences"]
    summaries: List[str] = []

    for i, text in enumerate(articles):
        if not text or len(text.strip()) == 0:
            summaries.append("요약할 텍스트가 없습니다.")
            continue

        print(
            f"기사 {i + 1}/{len(articles)}: LLM을 통해 뉴스 본문을 요약합니다. 요약 문장 개수: {num_sentences}개"
        )

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
    state["messages"].append(
        AIMessage(content=f"요약 완료: {len(summaries)}개 기사 요약")
    )
    # 요약 완료 표시
    state["next"] = "summarized"
    return state


def response_generator_agent_node(state: AgentState) -> AgentState:
    """Response Generator Agent 노드"""
    query = state["query"]
    summaries = state["summaries"]

    if not summaries:
        state["final_response"] = f"'{query}' 관련 뉴스를 찾을 수 없었습니다."
        return state

    # 요약들을 하나의 텍스트로 결합
    combined_summaries = "\n\n".join(
        [f"기사 {i + 1}:\n{summary}" for i, summary in enumerate(summaries)]
    )

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

    state["messages"].append(AIMessage(content="최종 응답 생성 완료"))
    # 응답 생성 완료 표시
    state["next"] = "completed"
    return state


def general_chat_agent_node(state: AgentState) -> AgentState:
    """General Chat Agent 노드"""
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

    state["messages"].append(AIMessage(content="일반 대화 응답 완료"))
    # 일반 대화 완료 표시
    state["next"] = "completed"
    return state


def routing_function(state: AgentState) -> str:
    """다음에 실행할 Agent를 결정하는 라우팅 함수"""
    next_agent = state.get("next", "")

    print(f"🔀 [ROUTING] next={next_agent}")

    if next_agent == "FINISH":
        return END
    elif next_agent in [
        "classifier_agent",
        "search_agent",
        "scraper_agent",
        "summarizer_agent",
        "response_generator_agent",
        "general_chat_agent",
    ]:
        return next_agent
    else:
        # 중간 상태들("classified", "searched", "scraped", "summarized", "completed")이나
        # 초기 상태("")는 모두 supervisor로
        return "supervisor"


def create_multiagent_graph() -> CompiledGraph:
    """
    Multi-Agent 뉴스 스크래핑 StateGraph를 생성합니다.

    Returns:
        CompiledGraph: 컴파일된 StateGraph
    """
    # StateGraph 생성
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("classifier_agent", classifier_agent_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("scraper_agent", scraper_agent_node)
    workflow.add_node("summarizer_agent", summarizer_agent_node)
    workflow.add_node("response_generator_agent", response_generator_agent_node)
    workflow.add_node("general_chat_agent", general_chat_agent_node)

    # 엣지 추가 (모든 Agent는 Supervisor를 거쳐서 라우팅)
    workflow.set_entry_point("supervisor")

    # Supervisor에서 각 Agent로 라우팅
    workflow.add_conditional_edges(
        "supervisor",
        routing_function,
        {
            "classifier_agent": "classifier_agent",
            "search_agent": "search_agent",
            "scraper_agent": "scraper_agent",
            "summarizer_agent": "summarizer_agent",
            "response_generator_agent": "response_generator_agent",
            "general_chat_agent": "general_chat_agent",
            END: END,
        },
    )

    # 각 Agent 실행 후 다시 Supervisor로 돌아감
    for agent in [
        "classifier_agent",
        "search_agent",
        "scraper_agent",
        "summarizer_agent",
        "response_generator_agent",
        "general_chat_agent",
    ]:
        workflow.add_edge(agent, "supervisor")

    # 그래프 컴파일
    return workflow.compile()


def visualize_graph(graph: CompiledGraph) -> None:
    """
    Multi-Agent StateGraph의 구조를 다양한 방식으로 시각화합니다.

    Args:
        graph: 컴파일된 StateGraph 인스턴스
    """
    print("\n📊 [Multi-Agent 그래프 구조 시각화]")
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
        print("- Supervisor가 중앙에서 모든 Agent들을 조정하는 구조입니다.")

    except Exception as e:
        print(f"❌ 그래프 시각화 중 오류 발생: {e}")
        print("LangGraph 버전이나 의존성을 확인해주세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Agent 기반 뉴스 스크래핑 시스템"
    )
    parser.add_argument(
        "--query", type=str, default="인공지능 최신 동향", help="검색할 키워드"
    )
    parser.add_argument(
        "--num_results", type=int, default=3, help="검색할 뉴스 기사 개수"
    )
    parser.add_argument("--num_sentences", type=int, default=3, help="요약할 문장 개수")
    parser.add_argument(
        "--visualize", action="store_true", help="그래프 구조를 시각화하고 종료"
    )
    args = parser.parse_args()

    # Multi-Agent StateGraph 생성
    graph = create_multiagent_graph()

    # 시각화 모드인 경우 그래프만 출력하고 종료
    if args.visualize:
        visualize_graph(graph)
        print("\n🎯 Multi-Agent 그래프 시각화가 완료되었습니다.")
        exit(0)

    # 초기 상태 설정
    initial_state = AgentState(
        query=args.query,
        messages=[],
        next="",
        is_news_related=False,
        news_urls=[],
        articles=[],
        summaries=[],
        final_response="",
        num_results=args.num_results,
        num_sentences=args.num_sentences,
    )

    print("🚀 Multi-Agent 시스템 실행")
    print(
        f"검색 키워드: '{args.query}', 기사 개수: {args.num_results}, 요약 문장 수: {args.num_sentences}"
    )
    print("=" * 80)

    # callback handler 생성
    debug_callback = MultiAgentDebugCallback()

    # 그래프 실행 (callback 포함)
    result = graph.invoke(initial_state, config={"callbacks": [debug_callback]})

    print("\n🎯 [최종 결과]")
    print("=" * 80)
    print(result["final_response"])

    print("\n📊 [실행 통계]")
    print(f"총 메시지 교환: {len(result['messages'])}개")
    print(f"검색된 URL: {len(result['news_urls'])}개")
    print(f"추출된 기사: {len(result['articles'])}개")
    print(f"생성된 요약: {len(result['summaries'])}개")
