# Text Analysis and Visualization with Streamlit

이 프로젝트는 스트림릿(Streamlit)을 활용하여 기본적인 텍스트 분석과 시각화를 지원하기 위해 설계되었습니다.
이를 통해 텍스트 데이터를 분석하고 단어 빈도 분석, 워드 클라우드 생성, 그리고 텍스트 내 주제의 네트워크 그래프 시각화를 수행할 수 있는 사용자 친화적인 인터페이스를 제공합니다.

## Table of Contents
- [Getting Started](#getting-started)
- [Description](#description)
- [Features](#features)
- [설치](#설치)
- [사용법](#사용법)
- [기여하기](#기여하기)
- [라이선스](#라이선스)
- [문의](#문의)

## Getting Started
이 프로젝트를 사용하기 전에 Python과 필요한 라이브러리가 설치되어 있는지 확인하십시오. 
필요한 라이브러리는 `requirements.txt` 파일을 사용하여 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

## Description
텍스트 분석, 단어 빈도 분석, 워드 클라우드 생성 및 LDA (Latent Dirichlet Allocation)를 사용하여 텍스트 내 주제의 네트워크 그래프를 시각화하는 데 중점을 둡니다.

## Features
### 1. 데이터 준비
  - 텍스트 데이터를 포함하는 CSV 또는 엑셀 파일을 업로드합니다.
  - 분석할 열을 선택합니다.

### 2. 분석 결과 표시
  - 데이터 요약과 가장 빈번한 단어를 포함한 데이터 요약을 확인합니다.
  - 단어 빈도 분석 결과를 CSV 파일로 다운로드할 수 있습니다.

### 3. 시각화
  - 바 차트로 상위 단어와 해당 빈도를 시각화합니다.
  - 단어 빈도를 그래픽 형식의 워드 클라우드로 생성합니다.
  - LDA를 사용하여 주제 및 관련 단어의 네트워크 그래프를 생성합니다.

## 설치
1. 이 저장소를 로컬 컴퓨터로 클론합니다.
``` bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

2. [Getting Started](#getting-started) 섹션에서 언급한 대로 필요한 종속성을 설치합니다.
3. Streamlit 앱을 실행합니다.


사용법
### 1. 데이터 준비
- 분석하려는 텍스트 데이터가 포함된 CSV 또는 엑셀 파일을 선택합니다.
- 지원하는 파일 형식인 경우 진행할 수 있고, 그렇지 않은 경우 오류 메시지가 표시됩니다.

### 2. 분석 결과 표시
- 데이터를 로드한 후, 데이터 요약 및 가장 빈번한 단어를 확인할 수 있습니다.
- 단어 빈도 분석 결과를 CSV 파일로 다운로드할 수 있습니다.

### 3. 시각화
- "시각화" 섹션에서 다양한 시각화 간에 전환할 수 있습니다.
  - Plot: 상위 단어 및 해당 빈도를 보여주는 바 차트를 확인합니다.
  - Word Cloud: 그래픽 형식으로 단어 빈도를 시각화하는 워드 클라우드를 생성합니다.
  - Network Graph: LDA를 사용하여 주제 및 관련 단어를 시각화하는 네트워크 그래프를 생성합니다.
 
## 라이선스
이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다. 자세한 내용은 LICENSE 파일을 참조하십시오.
