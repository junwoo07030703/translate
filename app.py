import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# ⚙️ 설정 및 상수
# ==========================================
st.set_page_config(page_title="AI 팟캐스트 번역기", page_icon="🎙️", layout="wide")

PRICE_WHISPER_PER_MIN = 0.006 
PRICE_GPT_INPUT_1M = 0.15
PRICE_GPT_OUTPUT_1M = 0.60
PRICE_TTS_1M_CHAR = 15.00
EXCHANGE_RATE = 1450 # 환율 약간 조정

# ==========================================
# 🛠️ 유틸리티 함수 (기존 로직 이식)
# ==========================================

def split_audio_file(file_path: Path, chunk_size_mb: int = 24) -> list[str]:
    """25MB 이상 파일 분할"""
    chunk_size = chunk_size_mb * 1024 * 1024
    file_size = file_path.stat().st_size
    
    if file_size <= chunk_size:
        return [str(file_path)]

    st.toast(f"✂️ 파일이 큽니다({file_size / (1024*1024):.1f}MB). 분할 처리를 시작합니다.", icon="✂️")
    chunk_files = []
    
    with open(file_path, 'rb') as f:
        part_num = 0
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            
            part_path = file_path.parent / f"{file_path.stem}_part{part_num}{file_path.suffix}"
            with open(part_path, 'wb') as chunk_f:
                chunk_f.write(chunk_data)
            
            chunk_files.append(str(part_path))
            part_num += 1
            
    return chunk_files

def transcribe_with_progress(client, audio_path, model="whisper-1"):
    """STT 수행 및 진행률 표시"""
    chunk_files = split_audio_file(Path(audio_path), chunk_size_mb=20)
    full_transcript = []
    
    progress_text = "음성 인식 중 (STT)..."
    my_bar = st.progress(0, text=progress_text)

    for idx, chunk_file in enumerate(chunk_files):
        with open(chunk_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text"
            )
            full_transcript.append(transcript)
        
        # 진행률 업데이트
        percent = int((idx + 1) / len(chunk_files) * 100)
        my_bar.progress(percent, text=f"{progress_text} ({idx+1}/{len(chunk_files)})")

        if len(chunk_files) > 1:
            os.remove(chunk_file)
            
    my_bar.empty()
    return " ".join(full_transcript)

def translate_long_text(text, model="gpt-4o-mini"):
    """번역 수행"""
    llm = ChatOpenAI(model=model, temperature=0, api_key=os.environ['OPENAI_API_KEY'])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate faithfully, preserve formatting. Do not add explanations."),
        ("human", "Translate the following from English to Korean:\n\n{chunk}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    translated_chunks = []
    my_bar = st.progress(0, text="번역 중 (GPT)...")
    
    for i, chunk in enumerate(chunks):
        out = chain.invoke({"chunk": chunk})
        translated_chunks.append(out)
        my_bar.progress(int((i + 1) / len(chunks) * 100))
        
    my_bar.empty()
    return "\n".join(translated_chunks)

def tts_chunked(text, client, model="tts-1", voice="nova"):
    """TTS 수행"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_text(text)
    
    temp_dir = Path(tempfile.gettempdir()) / "tts_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    mp3_files = []
    my_bar = st.progress(0, text="오디오 생성 중 (TTS)...")
    
    for i, chunk in enumerate(chunks):
        out_mp3 = temp_dir / f"part_{i:04d}_{int(time.time())}.mp3"
        response = client.audio.speech.create(model=model, voice=voice, input=chunk)
        response.stream_to_file(out_mp3)
        mp3_files.append(str(out_mp3))
        my_bar.progress(int((i + 1) / len(chunks) * 100))
        
    my_bar.empty()
    return mp3_files

def merge_mp3_simple(part_files):
    """MP3 병합 (Pure Python)"""
    if not part_files:
        return None
        
    merged_data = b""
    for p in part_files:
        p_path = Path(p)
        if p_path.exists():
            with open(p_path, "rb") as infile:
                merged_data += infile.read()
            # 병합 후 임시 파일 삭제
            try:
                os.remove(p_path)
            except:
                pass
                
    return merged_data

def display_cost_estimation(eng_text, ko_text):
    """비용 계산 및 UI 표시"""
    word_count = len(eng_text.split())
    est_duration_min = word_count / 150
    stt_cost = est_duration_min * PRICE_WHISPER_PER_MIN
    
    input_tokens = len(eng_text) / 4
    output_tokens = len(ko_text) / 1.5 
    trans_total = ((input_tokens / 1_000_000) * PRICE_GPT_INPUT_1M) + \
                  ((output_tokens / 1_000_000) * PRICE_GPT_OUTPUT_1M)
    
    tts_cost = (len(ko_text) / 1_000_000) * PRICE_TTS_1M_CHAR
    total_usd = stt_cost + trans_total + tts_cost
    total_krw = total_usd * EXCHANGE_RATE
    
    st.markdown("### 💰 예상 요금 명세서")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("STT (Whisper)", f"${stt_cost:.4f}", f"{est_duration_min:.1f}분")
    col2.metric("번역 (GPT)", f"${trans_total:.4f}", f"{int(output_tokens):,} 토큰")
    col3.metric("TTS (Audio)", f"${tts_cost:.4f}", f"{len(ko_text):,} 자")
    col4.metric("총 합계", f"${total_usd:.4f}", f"약 {int(total_krw):,}원")
    
    st.info("※ 위 금액은 추정치이며 실제 청구 금액과 다를 수 있습니다.")

# ==========================================
# 🖥️ Streamlit 메인 UI
# ==========================================

with st.sidebar:
    st.header("설정")
    api_key = st.text_input("OpenAI API Key", type="password", help="sk-로 시작하는 키를 입력하세요")
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        
    voice_option = st.selectbox("성우 선택 (TTS)", ["nova", "alloy", "echo", "fable", "onyx", "shimmer"])
    st.markdown("---")
    st.markdown("**사용 모델:**\n- STT: whisper-1\n- 번역: gpt-4o-mini\n- TTS: tts-1")

st.title("🎙️ 팟캐스트 AI 번역기")
st.markdown("영어 오디오 파일을 업로드하면 **한글 텍스트로 번역**하고 **한국어 오디오**로 만들어줍니다.")

uploaded_file = st.file_uploader("MP3 파일 업로드", type=["mp3"])

if uploaded_file and api_key:
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.audio(uploaded_file, format="audio/mp3")

    if st.button("🚀 번역 및 오디오 생성 시작", type="primary"):
        client = OpenAI(api_key=api_key)
        
        try:
            with st.status("작업 진행 중...", expanded=True) as status:
                # 1. STT
                st.write("🔊 음성을 텍스트로 변환 중 (STT)...")
                eng_text = transcribe_with_progress(client, tmp_file_path)
                st.write("✅ STT 완료!")
                
                # 2. 번역
                st.write("📝 한글로 번역 중 (GPT)...")
                ko_text = translate_long_text(eng_text)
                st.write("✅ 번역 완료!")
                
                # 3. 비용 계산
                display_cost_estimation(eng_text, ko_text)
                
                # 4. TTS
                st.write("🎙️ 한국어 오디오 생성 중 (TTS)...")
                mp3_parts = tts_chunked(ko_text, client, voice=voice_option)
                
                # 5. 병합
                st.write("💿 오디오 병합 중...")
                final_audio_bytes = merge_mp3_simple(mp3_parts)
                
                status.update(label="모든 작업이 완료되었습니다!", state="complete", expanded=False)

            # 결과 화면 표시
            st.divider()
            
            col_txt1, col_txt2 = st.columns(2)
            with col_txt1:
                with st.expander("영어 원문 (Transcript)"):
                    st.text_area("English", eng_text, height=300)
            with col_txt2:
                with st.expander("한글 번역 (Translation)"):
                    st.text_area("Korean", ko_text, height=300)
                    st.download_button("📜 번역 스크립트 다운로드", ko_text, file_name="script_ko.txt")

            st.subheader("🎧 최종 결과물")
            st.audio(final_audio_bytes, format="audio/mp3")
            
            st.download_button(
                label="📥 최종 MP3 다운로드",
                data=final_audio_bytes,
                file_name="translated_podcast.mp3",
                mime="audio/mp3",
                type="primary"
            )

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
        finally:
            # 원본 임시 파일 삭제
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

elif uploaded_file and not api_key:
    st.warning("👈 왼쪽 사이드바에 OpenAI API Key를 입력해주세요.")