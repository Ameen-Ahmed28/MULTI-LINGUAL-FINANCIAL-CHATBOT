let isRecording = false;
let recognition = null;
let currentLanguage = 'english';
let continuousListening = false;
let autoSendRecognized = true;
let speechTimeout = null;

// ENHANCED equation formatting function that actually works
function formatMessageContent(content) {
    console.log('Original content:', content);
    
    // STEP 1: Convert equations FIRST before other formatting
    content = convertEquationsToLatex(content);
    
    // STEP 2: Handle **bold** text (but preserve LaTeX)
    content = content.replace(/\*\*([^$]*?)\*\*/g, '<strong>$1</strong>');
    
    // STEP 3: Handle *italic* text  
    content = content.replace(/(?<!\*)\*([^*$]+)\*(?!\*)/g, '<em>$1</em>');
    
    // STEP 4: Handle numbered lists
    content = content.replace(/^(\d+)\.\s+(.+)$/gm, '<li><strong>$1.</strong> $2</li>');
    
    // STEP 5: Wrap consecutive list items in <ol>
    content = content.replace(/(<li>.*<\/li>\s*)+/g, function(match) {
        return '<ol>' + match + '</ol>';
    });
    
    // STEP 6: Handle bullet points
    content = content.replace(/^[•-]\s+(.+)$/gm, '<li>$1</li>');
    
    // STEP 7: Handle section headers
    content = content.replace(/^\*\*([^*$]+)\*\*$/gm, '<h3>$1</h3>');
    content = content.replace(/\*\*([^*$]*:)\*\*/g, '<h4>$1</h4>');
    
    // STEP 8: Convert line breaks
    content = content.replace(/\n\n+/g, '</p><p>');
    content = content.replace(/\n/g, '<br>');
    
    // STEP 9: Wrap in paragraphs
    if (!content.match(/^<(h[1-6]|ol|ul|div|p)/)) {
        content = '<p>' + content + '</p>';
    }
    
    // STEP 10: Clean up
    content = content.replace(/<p><\/p>/g, '');
    content = content.replace(/<p>\s*<\/p>/g, '');
    
    console.log('Formatted content:', content);
    return content;
}

// NEW FUNCTION: Convert specific equation patterns to LaTeX
function convertEquationsToLatex(content) {
    // Pattern 1: Handle Black-Scholes formula specifically
    content = content.replace(/C\s*=\s*S\s*N\(d1\)\s*-\s*K\s*e\^?\(?-rT\)?\s*N\(d2\)/gi, 
        '$$C = S \\cdot N(d_1) - K \\cdot e^{-rT} \\cdot N(d_2)$$');
    
    // Pattern 2: Handle Put option formula
    content = content.replace(/P\s*=\s*K\s*e\^?\(?-rT\)?\s*N\(-d2\)\s*-\s*S\s*N\(-d1\)/gi,
        '$$P = K \\cdot e^{-rT} \\cdot N(-d_2) - S \\cdot N(-d_1)$$');
    
    // Pattern 3: Handle d1 formula
    content = content.replace(/d1\s*:\s*d1\s*=\s*\(ln\(S\/K\)\s*\+\s*\(r\s*-\s*q\s*\+\s*\(σ\^2\/2\)\)\s*T\)\s*\/\s*\(σ\s*sqrt\(T\)\)/gi,
        '$$d_1 = \\frac{\\ln(S/K) + (r - q + \\frac{\\sigma^2}{2})T}{\\sigma\\sqrt{T}}$$');
    
    // Pattern 4: Handle d2 formula  
    content = content.replace(/d2\s*:\s*d2\s*=\s*d1\s*-\s*σ\s*sqrt\(T\)/gi,
        '$$d_2 = d_1 - \\sigma\\sqrt{T}$$');
    
    // Pattern 5: Handle PMT formula
    content = content.replace(/PMT\s*=\s*P\s*\[\s*i\(1\s*\+\s*i\)\^n\s*\]\s*\/\s*\[\s*\(1\s*\+\s*i\)\^n\s*-\s*1\s*\]/gi,
        '$$PMT = P \\cdot \\frac{i(1+i)^n}{(1+i)^n - 1}$$');
    
    // Pattern 6: Generic equation patterns
    // Handle any line that starts with a variable followed by = and contains math symbols
    content = content.replace(/^([A-Za-z_]+\d*)\s*[=:]\s*(.+[+\-*/()^√∑∏∫].+)$/gm, function(match, variable, equation) {
        // Skip if already in LaTeX format
        if (equation.includes('$$') || equation.includes('\\')) {
            return match;
        }
        
        // Convert the equation part to LaTeX
        let latexEq = convertToLatex(equation.trim());
        return `$$${variable} = ${latexEq}$$`;
    });
    
    return content;
}

// Helper function to convert common patterns to LaTeX
function convertToLatex(equation) {
    return equation
        // Handle fractions with brackets [a]/[b] -> \frac{a}{b}
        .replace(/\[([^\]]+)\]\s*\/\s*\[([^\]]+)\]/g, '\\frac{$1}{$2}')
        
        // Handle simple fractions a/b -> \frac{a}{b} (only if not already in LaTeX)
        .replace(/([^\\]|^)([a-zA-Z0-9\(\)]+)\s*\/\s*([a-zA-Z0-9\(\)]+)/g, '$1\\frac{$2}{$3}')
        
        // Handle exponents properly
        .replace(/\^(\w+)/g, '^{$1}')
        .replace(/\^(\([^)]+\))/g, '^{$1}')
        
        // Handle subscripts
        .replace(/([a-zA-Z])(\d+)/g, '$1_{$2}')
        
        // Handle square root
        .replace(/sqrt\(([^)]+)\)/g, '\\sqrt{$1}')
        
        // Handle natural log
        .replace(/ln\(([^)]+)\)/g, '\\ln($1)')
        
        // Handle exponential
        .replace(/e\^([^{])/g, 'e^{$1}')
        .replace(/exp\(([^)]+)\)/g, 'e^{$1}')
        
        // Handle Greek letters
        .replace(/sigma/gi, '\\sigma')
        .replace(/σ/g, '\\sigma')
        .replace(/alpha/gi, '\\alpha')
        .replace(/beta/gi, '\\beta')
        .replace(/gamma/gi, '\\gamma')
        .replace(/delta/gi, '\\delta')
        
        // Handle multiplication
        .replace(/\s\*\s/g, ' \\cdot ')
        
        // Handle common functions
        .replace(/N\(([^)]+)\)/g, 'N($1)')
        
        .trim();
}



// Initialize speech recognition
function initializeSpeechRecognition() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;
        
        recognition.onstart = function() {
            console.log('Speech recognition started');
            isRecording = true;
            document.getElementById('micBtn').classList.add('recording');
            document.getElementById('recordingIndicator').style.display = 'block';
            document.getElementById('speechStatusBar').style.display = 'flex';
            document.getElementById('speechStatusText').textContent = 'Listening...';
            document.getElementById('speechIndicator').textContent = '🔴';
        };
        
        recognition.onresult = function(event) {
            let finalTranscript = '';
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            // Show interim results in input field
            if (interimTranscript) {
                document.getElementById('messageInput').value = interimTranscript;
                document.getElementById('speechStatusText').textContent = 'Recognizing: ' + interimTranscript;
            }
            
            // Process final results
            if (finalTranscript) {
                document.getElementById('messageInput').value = finalTranscript;
                document.getElementById('speechStatusText').textContent = 'Recognized: ' + finalTranscript;
                
                // Auto-send logic
                if (autoSendRecognized) {
                    speechTimeout = setTimeout(() => {
                        if (finalTranscript.trim().length > 2) {
                            sendMessage();
                        }
                    }, 1500);
                }
            }
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
            document.getElementById('speechStatusText').textContent = 'Speech error: ' + event.error;
            
            if (event.error === 'not-allowed') {
                alert('Please allow microphone access in your browser settings and reload the page.');
            } else if (event.error === 'no-speech') {
                document.getElementById('speechStatusText').textContent = 'No speech detected. Try again.';
            }
            
            stopRecording();
        };
        
        recognition.onend = function() {
            console.log('Speech recognition ended');
            isRecording = false;
            document.getElementById('micBtn').classList.remove('recording');
            document.getElementById('recordingIndicator').style.display = 'none';
            
            // Restart if continuous listening is enabled
            if (continuousListening && document.getElementById('continuousListening').checked) {
                setTimeout(() => {
                    startRecording();
                }, 500);
            } else {
                setTimeout(() => {
                    document.getElementById('speechStatusBar').style.display = 'none';
                }, 2000);
                document.getElementById('speechIndicator').textContent = '🎤';
            }
        };
        
        updateSpeechLanguage();
        return true;
    } else {
        console.error('Speech recognition not supported');
        alert('Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.');
        return false;
    }
}

// Extended language mapping for all Indian languages
const speechLanguageMap = {
    'english': 'en-US',
    'hindi': 'hi-IN',
    'marathi': 'mr-IN', 
    'tamil': 'ta-IN',
    'bengali': 'bn-IN',
    'gujarati': 'gu-IN',
    'kannada': 'kn-IN',
    'malayalam': 'ml-IN',
    'punjabi': 'pa-IN',
    'telugu': 'te-IN',
    'urdu': 'ur-IN',
    'odia': 'or-IN',
    'assamese': 'as-IN',
    'nepali': 'ne-NP',
    'sindhi': 'sd-IN',
    'kashmiri': 'ks-IN',
    'sanskrit': 'sa-IN',
    'maithili': 'mai-IN',
    'dogri': 'doi-IN',
    'manipuri': 'mni-IN',
    'bodo': 'brx-IN',
    'santhali': 'sat-IN',
    'konkani': 'gom-IN'
};

function updateSpeechLanguage() {
    if (recognition) {
        recognition.lang = speechLanguageMap[currentLanguage] || 'en-US';
        console.log('Speech language updated to:', recognition.lang);
    }
}

function startRecording() {
    if (!recognition) {
        if (!initializeSpeechRecognition()) {
            return;
        }
    }
    
    try {
        // Stop any ongoing recognition first
        if (isRecording) {
            recognition.stop();
            setTimeout(() => {
                startRecording();
            }, 100);
            return;
        }
        
        updateSpeechLanguage();
        recognition.start();
        console.log('Starting speech recognition for language:', currentLanguage);
    } catch (error) {
        console.error('Error starting recognition:', error);
    }
}

function stopRecording() {
    if (recognition && isRecording) {
        recognition.stop();
    }
    
    if (speechTimeout) {
        clearTimeout(speechTimeout);
        speechTimeout = null;
    }
}

// Toggle recording
function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

// Clear chat function
async function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        try {
            const response = await fetch('/clear');
            const data = await response.json();
            
            if (data.success) {
                // Clear UI
                document.getElementById('chatMessages').innerHTML = `
                    <div class="welcome-message">
                        <div class="bot-message">
                            <div class="message-content">
                                Welcome! I'm your **specialized Financial AI Assistant**. I can help you with:
                                <br><br>• **Stocks, bonds, and investments**
                                <br>• **Banking and loans**  
                                <br>• **Financial planning and budgeting**
                                <br>• **Market trends and analysis**
                                <br>• **Options pricing (Black-Scholes)**
                                <br>• **Statistical analysis for market prediction**
                                <br><br>Please ask me any **finance-related questions** using text or voice!
                            </div>
                        </div>
                    </div>
                `;
                
                // Show success message
                showNotification('🗑️ Chat history cleared!', 'success');
            }
        } catch (error) {
            console.error('Error clearing chat:', error);
            showNotification('❌ Error clearing chat history', 'error');
        }
    }
}

// Show notification function
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        z-index: 10000;
        font-size: 14px;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (document.body.contains(notification)) {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }
    }, 4000);
}

// Toggle settings panel
function toggleSettings() {
    const panel = document.getElementById('settingsPanel');
    if (panel) {
        panel.style.display = panel.style.display === 'block' ? 'none' : 'block';
    }
}

// Handle language change
document.getElementById('languageSelect').addEventListener('change', function(e) {
    currentLanguage = e.target.value;
    updateSpeechLanguage();
    console.log('Language changed to:', currentLanguage);
});

// Handle continuous listening toggle
document.getElementById('continuousListening').addEventListener('change', function(e) {
    continuousListening = e.target.checked;
    
    if (continuousListening) {
        if (!isRecording) {
            startRecording();
        }
        showNotification('🔄 Continuous listening enabled', 'success');
    } else {
        stopRecording();
        showNotification('⏹️ Continuous listening disabled', 'info');
    }
});

// Handle auto-send toggle
document.getElementById('autoSendRecognized').addEventListener('change', function(e) {
    autoSendRecognized = e.target.checked;
    showNotification(
        autoSendRecognized ? '📤 Auto-send enabled' : '✋ Manual send mode', 
        'info'
    );
});

// Handle key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Send message
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Clear input
    messageInput.value = '';
    
    // Add user message to chat
    addMessage('user', message);
    
    // Show loading indicator
    const loadingIndicator = document.getElementById('loadingIndicator');
    if (loadingIndicator) {
        loadingIndicator.style.display = 'flex';
    }
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                language: currentLanguage
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addMessage('assistant', data.message);
            
            // Auto-play if enabled
            if (document.getElementById('autoPlayAudio')?.checked) {
                playAudio(data.message);
            }
        } else {
            addMessage('assistant', 'Sorry, I encountered an error processing your message.');
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('assistant', 'Sorry, I encountered a network error. Please try again.');
    } finally {
        // Hide loading indicator
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }
}

// Updated addMessage function with proper MathJax rendering
function addMessage(role, content) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    
    if (role === 'user') {
        messageDiv.innerHTML = `
            <div class="message-content">${content}</div>
        `;
    } else {
        // Format the content for better display
        const formattedContent = formatMessageContent(content);
        
        messageDiv.innerHTML = `
            <div class="message-content">${formattedContent}</div>
            <div class="message-actions">
                <button class="action-btn" onclick="copyText('${messageId}')" title="Copy message">
                    📋 Copy
                </button>
                <button class="action-btn" onclick="playAudio('${messageId}')" title="Play audio">
                    🔊 Play
                </button>
            </div>
        `;
    }
    
    messageDiv.setAttribute('data-message-id', messageId);
    messageDiv.setAttribute('data-content', content);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // ENSURE MATHJAX RENDERS THE NEW CONTENT
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([messageDiv]).then(() => {
            console.log('✅ MathJax rendered successfully');
        }).catch((err) => {
            console.error('❌ MathJax error:', err);
        });
    } else {
        console.warn('⚠️ MathJax not available');
    }
}


// Updated copy function
function copyText(messageId) {
    const messageDiv = document.querySelector(`[data-message-id="${messageId}"]`);
    const content = messageDiv.getAttribute('data-content');
    
    navigator.clipboard.writeText(content).then(() => {
        showNotification('📋 Copied to clipboard!', 'success');
        
        // Visual feedback on button
        const copyBtn = messageDiv.querySelector('.action-btn');
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '✅ Copied';
        copyBtn.style.background = '#10b981';
        copyBtn.style.color = 'white';
        
        setTimeout(() => {
            copyBtn.innerHTML = originalText;
            copyBtn.style.background = '';
            copyBtn.style.color = '';
        }, 2000);
    }).catch(err => {
        console.error('Copy failed:', err);
        showNotification('❌ Copy failed', 'error');
    });
}

// Updated audio function
function playAudio(messageId) {
    const messageDiv = document.querySelector(`[data-message-id="${messageId}"]`);
    const content = messageDiv.getAttribute('data-content');
    
    // Show loading state
    const audioBtn = messageDiv.querySelector('.action-btn[onclick*="playAudio"]');
    const originalText = audioBtn.innerHTML;
    audioBtn.innerHTML = '🔄 Loading...';
    audioBtn.disabled = true;
    
    fetch('/tts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: content,
            language: currentLanguage
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const audio = new Audio(data.audio_url);
            
            // Update button during playback
            audioBtn.innerHTML = '⏸️ Playing...';
            
            audio.onended = () => {
                audioBtn.innerHTML = originalText;
                audioBtn.disabled = false;
            };
            
            audio.onerror = () => {
                audioBtn.innerHTML = originalText;
                audioBtn.disabled = false;
                showNotification('❌ Audio playback failed', 'error');
            };
            
            audio.play();
        } else {
            throw new Error('TTS failed');
        }
    })
    .catch(error => {
        console.error('TTS Error:', error);
        audioBtn.innerHTML = originalText;
        audioBtn.disabled = false;
        showNotification('❌ Audio generation failed', 'error');
    });
}

// Load chat history and initialize on page load
window.addEventListener('load', async function() {
    console.log('Page loaded, initializing...');
    
    // Initialize speech recognition
    setTimeout(() => {
        initializeSpeechRecognition();
    }, 1000);
    
    try {
        const response = await fetch('/history');
        const history = await response.json();
        
        // Clear welcome message if there's history
        if (history.length > 0) {
            document.getElementById('chatMessages').innerHTML = '';
            
            history.forEach(msg => {
                addMessage(msg.role, msg.content);
            });
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
});

// Close settings when clicking outside
document.addEventListener('click', function(event) {
    const settingsPanel = document.getElementById('settingsPanel');
    const settingsBtn = document.querySelector('.settings-btn');
    
    if (settingsPanel && settingsBtn && 
        !settingsPanel.contains(event.target) && 
        !settingsBtn.contains(event.target)) {
        settingsPanel.style.display = 'none';
    }
});

// Handle page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.hidden && isRecording && !continuousListening) {
        stopRecording();
    }
});
