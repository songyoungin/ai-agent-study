"""
검색 키워드로부터 뉴스 기사를 검색하고, 기사 본문을 추출하고, 요약하는 에이전트를 생성합니다.

실행 방법:
    python web_scraping_agent.py --query "인공지능 최신 동향" --num_results 5 --num_sentences 5

세부 구현 사항:
    - 뉴스 기사 검색은 DuckDuckGo API를 사용합니다.
    - 기사 본문 추출은 newspaper 라이브러리를 사용합니다.
    - 요약은 OpenAI API를 사용합니다.
    - LangGraph를 사용해 에이전트를 생성합니다.
"""

import argparse

from typing import List
from langgraph.graph.graph import CompiledGraph
from newspaper import Article
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


from dotenv import load_dotenv

load_dotenv(verbose=True)

# OpenAI 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-4o", temperature=0)


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
        f"검색 키워드: [{query}]에 대해 DuckDuckGo API를 이용해 기사 검색을 시작합니다. 최대 검색 결과: {num_results}개"
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

    # num_results가 바인딩된 search_news 도구 생성
    @tool
    def search_news(query: str) -> List[str]:
        """DuckDuckGo를 사용해 키워드로 뉴스 기사 URL을 검색합니다."""
        return search_news_base(query, num_results)

    # 기본 fetch_article 도구 생성
    @tool
    def fetch_article(url: str) -> str:
        """주어진 URL의 뉴스 본문을 스크래핑합니다."""
        return fetch_article_base(url)

    # num_sentences가 바인딩된 summarize_text 도구 생성
    @tool
    def summarize_text(text: str) -> str:
        """긴 텍스트를 지정된 문장 수로 요약합니다."""
        return summarize_text_base(text, num_sentences)

    # 바인딩된 도구들로 에이전트 생성
    return create_react_agent(llm, tools=[search_news, fetch_article, summarize_text])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="뉴스 기사 검색 및 요약 에이전트")
    parser.add_argument(
        "--query", type=str, default="인공지능 최신 동향", help="검색할 키워드"
    )
    parser.add_argument(
        "--num_results", type=int, default=3, help="검색할 뉴스 기사 개수"
    )
    parser.add_argument("--num_sentences", type=int, default=3, help="요약할 문장 개수")
    args = parser.parse_args()

    # 매개변수가 바인딩된 에이전트 생성
    agent = create_agent_with_params(args.num_results, args.num_sentences)

    # 태스크 생성
    task = f"'{args.query}' 관련 최신 뉴스를 찾아서 각 기사를 요약해줘. 검색된 모든 기사를 처리해야 해."

    print(
        f"실행 설정: 검색 키워드='{args.query}', 기사 개수={args.num_results}, 요약 문장 수={args.num_sentences}"
    )
    print("=" * 80)

    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    print(result["messages"][-1].content)
