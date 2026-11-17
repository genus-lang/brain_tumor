/**
 * care_assistant.js - NeuroScan AI Patient Care Assistant Client Logic
 * Manages tabs, live chatbot communication with /api/chat, typing indicators,
 * and structured markdown-to-HTML rendering.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. TAB NAVIGATION
    const tabButtons = document.querySelectorAll(".care-tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            if (!targetId) return;

            tabButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }
        });
    });

    // 2. CHATBOT ENGINE
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatContext = document.getElementById("chatContext");
    const chatMessages = document.getElementById("chatMessages");
    const chatSendBtn = document.getElementById("chatSendBtn");
    const suggestionChips = document.getElementById("suggestionChips");

    // Markdown simple parser for clean chat formatting
    function formatMarkdown(text) {
        if (!text) return "";

        // Headers
        let formatted = text
            .replace(/^### (.*$)/gim, '<h4>$1</h4>')
            .replace(/^#### (.*$)/gim, '<strong>$1</strong>');

        // Bold & Italic
        formatted = formatted
            .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/gim, '<em>$1</em>');

        // Blockquotes & Alerts
        formatted = formatted
            .replace(/^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*$)/gim, '<div class="alert-inline alert-$1"><strong>$1:</strong> $2</div>')
            .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

        // Lists
        formatted = formatted.replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');
        formatted = formatted.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');
        formatted = formatted.replace(/<\/ul>\s*<ul>/gim, '');

        // Paragraphs
        const paragraphs = formatted.split(/\n\n+/);
        formatted = paragraphs.map(p => {
            p = p.trim();
            if (!p.startsWith('<h') && !p.startsWith('<ul') && !p.startsWith('<div') && !p.startsWith('<block')) {
                return `<p>${p}</p>`;
            }
            return p;
        }).join('');

        return formatted;
    }

    // Helper: append message bubble
    function appendMessage(sender, htmlContent) {
        if (!chatMessages) return;

        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-msg ${sender}-msg`;

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const avatarIcon = sender === "bot" ? '<i class="fa-solid fa-user-doctor"></i>' : '<i class="fa-solid fa-user"></i>';

        msgDiv.innerHTML = `
            <div class="msg-avatar">${avatarIcon}</div>
            <div class="msg-bubble">
                <div class="msg-text">${htmlContent}</div>
                <span class="msg-time">${timeStr}</span>
            </div>
        `;

        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Helper: show typing indicator
    function showTypingIndicator() {
        const typingDiv = document.createElement("div");
        typingDiv.className = "chat-msg bot-msg typing-msg";
        typingDiv.id = "typingIndicator";
        typingDiv.innerHTML = `
            <div class="msg-avatar"><i class="fa-solid fa-user-doctor"></i></div>
            <div class="msg-bubble">
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeTypingIndicator() {
        const ind = document.getElementById("typingIndicator");
        if (ind) ind.remove();
    }

    // Send query to /api/chat
    async function handleSend(userText) {
        if (!userText || !userText.trim()) return;

        const cleanText = userText.trim();
        appendMessage("user", `<p>${cleanText.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</p>`);

        if (chatInput) chatInput.value = "";
        if (chatSendBtn) chatSendBtn.disabled = true;

        showTypingIndicator();

        try {
            const contextVal = chatContext ? chatContext.value : "";
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: cleanText,
                    context: contextVal
                })
            });

            removeTypingIndicator();

            if (!res.ok) {
                throw new Error(`Server returned status ${res.status}`);
            }

            const data = await res.json();
            const formattedReply = formatMarkdown(data.reply);

            let footerHTML = "";
            if (data.sources && data.sources.length) {
                footerHTML += `<div class="msg-disclaimer"><i class="fa-solid fa-book-medical me-1 text-primary"></i> <strong>Sources:</strong> ${data.sources.join(" &bull; ")}</div>`;
            }

            appendMessage("bot", formattedReply + footerHTML);

            // Update suggestion chips with new follow-up questions if provided
            if (data.suggested_followups && suggestionChips) {
                suggestionChips.innerHTML = "";
                data.suggested_followups.forEach(q => {
                    const btn = document.createElement("button");
                    btn.type = "button";
                    btn.className = "chip-btn";
                    btn.setAttribute("data-query", q);
                    btn.textContent = q;
                    btn.addEventListener("click", () => {
                        handleSend(q);
                    });
                    suggestionChips.appendChild(btn);
                });
            }

        } catch (err) {
            removeTypingIndicator();
            appendMessage("bot", `<p><i class="fa-solid fa-triangle-exclamation text-danger me-1"></i> I encountered an error retrieving this guidance. Please consult a healthcare professional or try asking in different words.</p>`);
            console.error("Chat API error:", err);
        } finally {
            if (chatSendBtn) chatSendBtn.disabled = false;
            if (chatInput) chatInput.focus();
        }
    }

    if (chatForm) {
        chatForm.addEventListener("submit", e => {
            e.preventDefault();
            if (chatInput) {
                handleSend(chatInput.value);
            }
        });
    }

    // Suggested Question Chips
    if (suggestionChips) {
        suggestionChips.addEventListener("click", e => {
            const btn = e.target.closest(".chip-btn");
            if (btn) {
                const query = btn.getAttribute("data-query");
                if (query) {
                    // Switch to chat tab if not active
                    const chatTabBtn = document.querySelector('[data-tab="tab-chat"]');
                    if (chatTabBtn && !chatTabBtn.classList.contains("active")) {
                        chatTabBtn.click();
                    }
                    handleSend(query);
                }
            }
        });
    }
});

