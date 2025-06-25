"""
웹 스크래핑 & 요약 에이전트 (LangGraph 활용)
"""

from typing import List
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


@tool
def search_news(query: str, num_results: int = 3) -> List[str]:
    """
    DuckDuckGo를 사용해 키워드로 뉴스 기사 URL을 검색합니다.

    Args:
        query: 검색 키워드
        num_results: 반환할 URL 개수 (기본값: 3)

    Returns:
        List[str]: 검색된 URL 리스트
    """
    print(
        f"검색 키워드: [{query}]에 대해 DuckDuckGo API를 이용해 기사 검색을 시작합니다."
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


@tool
def fetch_article(url: str) -> str:
    """
    주어진 URL의 뉴스 본문을 스크래핑합니다.

    Args:
        url: 뉴스 기사 URL

    Returns:
        str: 본문 텍스트 (에러 시 에러 메시지)
    """
    try:
        article = Article(url)
        article.download()
        article.parse()

        if not article.text:
            return f"URL에서 본문을 추출할 수 없습니다: {url}"

        return article.text
    except Exception as e:  # newspaper 라이브러리 예외 처리
        return f"기사 추출 중 오류 발생: {str(e)}"


@tool
def summarize_text(text: str) -> str:
    """
    긴 텍스트를 3문장 이내로 요약합니다.

    Args:
        text: 원본 텍스트

    Returns:
        str: 요약된 텍스트
    """
    if not text or len(text.strip()) == 0:
        return "요약할 텍스트가 없습니다."

    prompt = f"다음 뉴스 본문을 3문장 이내로 한국어로 요약해 주세요:\n\n{text}\n\n요약:"

    try:
        response = llm.invoke([HumanMessage(content=prompt)])

        # response.content가 문자열이 아닌 경우 처리
        if isinstance(response.content, str):
            return response.content
        else:
            return str(response.content)
    except Exception as e:
        return f"요약 중 오류 발생: {str(e)}"


# LangGraph를 사용해 에이전트 생성
agent = create_react_agent(llm, tools=[search_news, fetch_article, summarize_text])


if __name__ == "__main__":
    user_query = "인공지능 최신 동향"
    task = f"'{user_query}' 관련 최신 뉴스 3개를 찾아 각각 3문장으로 요약해줘"
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    print(result["messages"][-1].content)
