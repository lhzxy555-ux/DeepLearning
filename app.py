import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 下载停用词（首次运行需要）
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 文本清洗函数（与 notebook 中保持一致）
def clean_text_advanced(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# 加载模型和 tokenizer
@st.cache_resource  # 缓存，避免重复加载
def load_models():
    model = load_model('imdb_lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

st.title("🎬 IMDB 影评情感分析")
st.write("输入英文影评，模型将判断情感为正面还是负面")

model, tokenizer = load_models()

review = st.text_area("输入影评文字：", height=150)

if st.button("🔍 分析情感"):
    if review.strip():
        # 清洗和预测
        cleaned = clean_text_advanced(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
        pred = model.predict(padded)[0][0]
        
        # 显示结果
        sentiment = "😊 正面" if pred > 0.5 else "😞 负面"
        confidence = pred if pred > 0.5 else 1 - pred
        
        st.success(f"**情感分析结果：{sentiment}**")
        st.info(f"置信度：{confidence:.4f}")
        
        # 显示清洗后的文本（可选）
        with st.expander("查看处理后的文本"):
            st.write(cleaned)
    else:
        st.warning("请输入影评内容")