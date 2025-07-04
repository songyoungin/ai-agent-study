"""검색 키워드로부터 뉴스 기사를 검색하고, 기사 본문을 추출 후 요약하는 에이전트를 생성합니다.

실행 방법:
    뉴스 기사 검색 예시: python web_scraping_agent.py --query "인공지능 최신 동향" --num_results 5 --num_sentences 5
    일상 대화 예시: python web_scraping_agent.py --query "오늘 날씨 어때?"

세부 구현 사항:
    - 뉴스 기사 검색은 DuckDuckGo API를 사용합니다.
    - 기사 본문 추출은 newspaper 라이브러리를 사용합니다.
    - 요약은 OpenAI API를 사용합니다.
    - LangGraph를 사용해 에이전트를 생성합니다.
    - 뉴스와 관련 없는 쿼리는 뉴스 검색을 하지 않습니다.
"""

import argparse
from typing import List, Optional

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.tracers.context import collect_runs
from langchain_openai import ChatOpenAI
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langsmith.run_trees import RunTree
from newspaper import Article

load_dotenv(verbose=True)

# OpenAI 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)


def is_news_related_query(query: Optional[str]) -> bool:
    """
    쿼리가 뉴스 검색과 관련이 있는지 판단합니다.

    Args:
        query: 분석할 쿼리 문자열

    Returns:
        bool: 뉴스 검색이 필요하면 True, 아니면 False
    """
    if not query or not query.strip():
        return False

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
            return True
        elif answer.startswith("아니오") or answer.startswith("no"):
            return False
        else:
            # 명확하지 않은 경우 기본적으로 뉴스 검색 수행
            return True

    except Exception as e:
        print(f"쿼리 분류 중 오류 발생: {e}")
        # 오류 발생 시 안전하게 뉴스 검색 수행
        return True


def search_news_base(query: str, num_results: int = 3) -> List[str]:
    """
    DuckDuckGo를 사용해 키워드로 뉴스 기사 URL을 검색합니다.

    Args:
        query: 검색 키워드
        num_results: 반환할 URL 개수 (기본값: 3)

    Returns:
        List[str]: 검색된 URL 리스트
    """
    print(
        f"검색 키워드: [{query}]에 대해 DuckDuckGo API를 이용해 기사 검색을 시작합니다. 최대 검색 결과: {num_results}개"  # noqa: E501
    )

    ddgs = DDGS()
    urls: List[str] = []

    try:
        # DuckDuckGo API를 이용해 KR 리전의 뉴스 기사 검색 결과를 반환
        results = ddgs.news(query, region="kr-kr", max_results=num_results)
        print(f"검색 결과: {results}")

        for item in results:
            # DuckDuckGo API 검색 결과 중 기사 링크를 여러 URL 필드에서 추출
            url = item.get("url") or item.get("href") or item.get("link")
            if url and url.startswith("http"):
                urls.append(url)
                if len(urls) >= num_results:
                    break
    except Exception as e:
        print(f"DuckDuckGo API를 이용해 기사 검색 중 오류가 발생했습니다: {e}")

    return urls


def fetch_article_base(url: str) -> str:
    """
    주어진 URL의 뉴스 본문을 스크래핑합니다.

    Args:
        url: 뉴스 기사 URL

    Returns:
        str: 본문 텍스트 (에러 시 에러 메시지)
    """
    print(f"URL: {url}에 대해 뉴스 본문을 추출합니다.")

    try:
        article = Article(url)
        article.download()
        article.parse()

        if not article.text:
            return f"URL에서 본문을 추출할 수 없습니다: {url}"

        return article.text
    except Exception as e:  # newspaper 라이브러리 예외 처리
        return f"기사 추출 중 오류 발생: {str(e)}"


def summarize_text_base(text: str, num_sentences: int = 3) -> str:
    """
    긴 텍스트를 지정된 문장 수로 요약합니다.

    Args:
        text: 원본 텍스트
        num_sentences: 요약할 문장 개수 (기본값: 3)

    Returns:
        str: 요약된 텍스트
    """
    if not text or len(text.strip()) == 0:
        return "요약할 텍스트가 없습니다."

    print(f"LLM을 통해 뉴스 본문을 요약합니다. 요약 문장 개수: {num_sentences}개")

    prompt = f"다음 뉴스 본문을 {num_sentences}개의 문장 이내로 한국어로 요약해 주세요:\n\n{text}\n\n요약:"

    try:
        response = llm.invoke([HumanMessage(content=prompt)])

        # response.content가 문자열이 아닌 경우 처리
        if isinstance(response.content, str):
            return response.content
        else:
            return str(response.content)
    except Exception as e:
        return f"요약 중 오류 발생: {str(e)}"


def create_agent_with_params(num_results: int, num_sentences: int) -> CompiledGraph:
    """
    매개변수가 바인딩된 도구들로 에이전트를 생성합니다.

    Args:
        num_results: 검색할 뉴스 기사 개수
        num_sentences: 요약할 문장 개수

    Returns:
        CompiledGraph: LangGraph 에이전트
    """

    # 쿼리 분류 도구 생성
    @tool
    def classify_query(query: str) -> str:
        """사용자 쿼리가 뉴스 검색과 관련이 있는지 판단합니다."""
        is_news_related = is_news_related_query(query)
        if is_news_related:
            return "뉴스 검색이 필요한 쿼리입니다."
        else:
            return "뉴스 검색이 필요하지 않은 일반적인 질문입니다."

    # num_results가 바인딩된 search_news 도구 생성
    @tool
    def search_news(query: str) -> List[str]:
        """DuckDuckGo를 사용해 키워드로 뉴스 기사 URL을 검색합니다."""
        return search_news_base(query, num_results)

    # 기본 fetch_article 도구 생성
    @tool
    def fetch_article(url: str) -> str:
        """주어진 URL의 뉴스 본문을 스크래핁합니다."""
        return fetch_article_base(url)

    # num_sentences가 바인딩된 summarize_text 도구 생성
    @tool
    def summarize_text(text: str) -> str:
        """긴 텍스트를 지정된 문장 수로 요약합니다."""
        return summarize_text_base(text, num_sentences)

    # 바인딩된 도구들로 에이전트 생성
    return create_react_agent(llm, tools=[classify_query, search_news, fetch_article, summarize_text])


def print_run_tree(node: RunTree, level: int = 0) -> None:
    """Langchain 컬렉터에 의해 수집된 run의 정보를 재귀적으로 출력합니다.
    도구 실행(tool)과 주요 체인(chain) 실행만 출력하며, 메타데이터는 제외합니다.

    Args:
        node: 컬렉터에 의해 수집된 실행 결과
        level: 재귀 깊이 (기본값: 0)
    """
    # 도구 실행과 체인 실행만 출력
    if node.run_type in ["tool", "chain"]:
        indent = "  " * level
        print(f"{indent}- 이름: {node.name!r}, 타입: {node.run_type}")

        # 도구 실행인 경우 핵심 정보만 출력
        if node.run_type == "tool":
            if hasattr(node, "inputs") and node.inputs is not None:
                # 도구 입력에서 핵심 파라미터만 추출
                if isinstance(node.inputs, dict) and "input" in node.inputs:
                    print(f"{indent}  입력: {node.inputs['input']}")

            if hasattr(node, "outputs") and node.outputs is not None:
                # 도구 출력에서 content만 추출
                if isinstance(node.outputs, dict) and "output" in node.outputs:
                    output = node.outputs["output"]
                    if hasattr(output, "content"):
                        print(f"{indent}  출력: {output.content}")

        # 체인 실행인 경우 간소화된 정보만 출력
        elif node.run_type == "chain":
            if hasattr(node, "inputs") and node.inputs is not None:
                # 입력 길이 제한 (100자)
                inputs_str = str(node.inputs)
                if len(inputs_str) > 100:
                    inputs_str = inputs_str[:100] + "..."
                print(f"{indent}  입력: {inputs_str}")

        # 하위 호출 순회 (출력 조건을 만족하는 경우 레벨 증가)
        for child in getattr(node, "child_runs", []):
            print_run_tree(child, level + 1)
    else:
        # 출력하지 않지만 하위 호출은 계속 순회 (레벨 유지)
        for child in getattr(node, "child_runs", []):
            print_run_tree(child, level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="뉴스 기사 검색 및 요약 에이전트")
    parser.add_argument("--query", type=str, default="인공지능 최신 동향", help="검색할 키워드")
    parser.add_argument("--num_results", type=int, default=3, help="검색할 뉴스 기사 개수")
    parser.add_argument("--num_sentences", type=int, default=3, help="요약할 문장 개수")
    args = parser.parse_args()

    # 매개변수가 바인딩된 에이전트 생성
    agent = create_agent_with_params(args.num_results, args.num_sentences)

    # 태스크 생성
    task = f"""
    다음 사용자 쿼리를 처리해주세요: '{args.query}'

    처리 순서:
    1. 먼저 classify_query 도구를 사용해서 이 쿼리가 뉴스 검색과 관련이 있는지 판단하세요.

    2. 뉴스 검색이 필요한 쿼리인 경우:
    - search_news로 관련 뉴스 기사 URL들을 검색하세요
    - 각 URL에 대해 fetch_article로 기사 본문을 추출하세요
    - 각 기사를 summarize_text로 요약하세요
    - 모든 기사의 요약을 종합해서 사용자에게 제공하세요

    3. 뉴스 검색이 필요하지 않은 일반적인 질문인 경우:
    - 뉴스 검색 도구들을 사용하지 말고, 직접 친근하게 답변하세요
    - 예: 인사말에는 인사로 답하고, 일반적인 질문에는 도움이 되는 답변을 제공하세요

    항상 한국어로 답변하세요.
    """

    print(f"실행 설정: 검색 키워드='{args.query}', 기사 개수={args.num_results}, 요약 문장 수={args.num_sentences}")
    print("=" * 80)

    # 각 단계별 기록을 확인할 수 있는 인메모리 컬렉터 생성
    with collect_runs() as collector:
        result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    # 컬렉터 결과 출력
    for tree in collector.traced_runs:
        print_run_tree(tree)

    # 최종 답변 출력
    print(result["messages"][-1].content)
