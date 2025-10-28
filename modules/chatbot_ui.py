import streamlit as st

def chatbot_ui():
    """
    Floating chatbot with click-to-open functionality (no extra text area)
    """

    st.markdown(
        """
        <style>
        .chatbot-container {
            position: fixed;
            bottom: 20px;
            right: 25px;
            z-index: 9999;
        }

        .chatbot-icon {
            background-color: #FF4B4B;
            color: white;
            border-radius: 50%;
            padding: 14px 18px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }

        .chatbot-icon:hover {
            background-color: #e63946;
            transform: scale(1.1);
        }

        .chat-popup {
            display: none;
            position: fixed;
            bottom: 80px;
            right: 25px;
            width: 320px;
            height: 420px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            padding: 10px;
            overflow-y: auto;
            z-index: 10000;
        }

        .chat-popup.open {
            display: block;
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }

        .chat-header {
            font-weight: bold;
            color: #FF4B4B;
            margin-bottom: 10px;
        }

        .chat-body {
            height: 320px;
            overflow-y: auto;
            padding-right: 5px;
        }

        .user-msg {
            background-color: #DCF8C6;
            padding: 6px;
            border-radius: 8px;
            margin-bottom: 5px;
            text-align: right;
        }

        .bot-msg {
            background-color: #F1F0F0;
            padding: 6px;
            border-radius: 8px;
            margin-bottom: 5px;
            text-align: left;
        }

        .chat-input {
            width: 100%;
            padding: 6px;
            margin-top: 5px;
            border: 1px solid #ccc;
            border-radius: 8px;
        }
        </style>

        <div class="chatbot-container">
            <div class="chat-popup" id="chatbox">
                <div class="chat-header">🤖 AI Chat Assistant</div>
                <div class="chat-body" id="chatBody">
                    <!-- Messages will appear here -->
                </div>
                <input type="text" id="userInput" class="chat-input" placeholder="Type your question...">
            </div>
            <div class="chatbot-icon" id="chatbotIcon">💬</div>
        </div>

        <script>
        const chatbotIcon = document.getElementById("chatbotIcon");
        const chatbox = document.getElementById("chatbox");
        const chatBody = document.getElementById("chatBody");
        const userInput = document.getElementById("userInput");

        chatbotIcon.addEventListener("click", function() {
            chatbox.classList.toggle("open");
            if (chatbox.classList.contains("open") && chatBody.innerHTML.trim() === "") {
                chatBody.innerHTML = "<div class='bot-msg'>👋 Hi! I'm your assistant. Ask me about preprocessing, missing values, or correlation!</div>";
            }
        });

        userInput.addEventListener("keypress", function(event) {
            if (event.key === "Enter") {
                const userText = userInput.value.trim();
                if (userText !== "") {
                    chatBody.innerHTML += `<div class='user-msg'>${userText}</div>`;
                    userInput.value = "";

                    // Simple Q&A responses
                    let response = "";
                    const text = userText.toLowerCase();
                    if (text.includes("missing")) {
                        response = "🧩 Missing values occur when some data points are empty. You can fill them with mean, median, or mode.";
                    } else if (text.includes("outlier")) {
                        response = "📊 Outliers are extreme values. Use IQR or Z-score to handle them.";
                    } else if (text.includes("standard") || text.includes("scaling")) {
                        response = "⚖️ Standardization normalizes numeric data for better performance.";
                    } else if (text.includes("encode")) {
                        response = "🔠 Encoding converts categorical values into numbers using Label or One-Hot Encoding.";
                    } else if (text.includes("correlation")) {
                        response = "🔗 Correlation shows how variables are related. Check it with a heatmap!";
                    } else if (text.includes("thank")) {
                        response = "😊 You're welcome!";
                    } else {
                        response = "🤔 Try asking about missing values, outliers, encoding, or standardization.";
                    }

                    setTimeout(() => {
                        chatBody.innerHTML += `<div class='bot-msg'>${response}</div>`;
                        chatBody.scrollTop = chatBody.scrollHeight;
                    }, 500);
                }
            }
        });
        </script>
        """,
        unsafe_allow_html=True,
    )
