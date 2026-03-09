import requests
import streamlit as st
import os 

# 🔧 Адрес вашего бэкенда на RunPod (HTTPS прокси)
API_URL = os.getenv("API_URL", "https://lmaau3rvbv7tey-8088.proxy.runpod.net")

st.set_page_config(page_title="Классификатор", layout="centered")
st.title("🤖 Классификатор: Текст + Изображения")

tab1, tab2 = st.tabs(['📝 Текст', '🖼️ Изображение'])

# ===== Текст =====
with tab1:
    st.subheader("Классификация текста")
    txt = st.text_area("Введите текст для классификации", height=100)
    
    if st.button("Классифицировать текст", type="primary"):
        if txt.strip():
            with st.spinner("Обрабатываю..."):
                try:
                    response = requests.post(
                        f"{API_URL}/clf_text",
                        json={"text": txt},
                        timeout=30
                        # ✅ verify=False больше не нужен — сертификат валидный!
                    )
                    if response.status_code == 200:
                        res = response.json()
                        st.success("✅ Готово!")
                        st.write(f"**Текст:** {res['text']}")
                        st.write(f"**Класс:** `{res['class_name']}` (ID: {res['predicted_class']})")
                        st.write(f"**Уверенность:** {res['confidence']:.2%}")
                    else:
                        st.error(f"❌ Ошибка API: {response.status_code}\n{response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Не удалось подключиться к бэкенду")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
        else:
            st.warning("⚠️ Введите текст")

# ===== Изображение =====
with tab2:
    st.subheader("Детекция объектов на изображении")
    image = st.file_uploader("Загрузите изображение", type=['jpg', 'jpeg', 'png', 'webp'])
    
    if st.button("Обработать изображение", type="primary"):
        if image is not None:
            with st.spinner("Обрабатываю..."):
                try:
                    st.image(image, caption="Загруженное изображение", use_container_width=True)
                    
                    files = {"file": image.getvalue()}
                    response = requests.post(
                        f"{API_URL}/clf_image",
                        files=files,
                        timeout=60
                        # ✅ verify=False больше не нужен
                    )
                    
                    if response.status_code == 200:
                        res = response.json()
                        detections = res.get("detections", [])
                        
                        if detections:
                            st.success(f"✅ Найдено объектов: {len(detections)}")
                            for i, obj in enumerate(detections, 1):
                                st.write(f"**{i}.** `{obj['class']}` — {obj['confidence']:.2%}")
                        else:
                            st.info("ℹ️ Объекты не обнаружены")
                    else:
                        st.error(f"❌ Ошибка API: {response.status_code}\n{response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Не удалось подключиться к бэкенду")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
        else:
            st.warning("⚠️ Загрузите изображение")

st.markdown("---")
st.caption("Бэкенд: FastAPI + RuBERT + YOLO | Фронтенд: Streamlit")