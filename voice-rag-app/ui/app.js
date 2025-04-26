// ui/app.js
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const pdfUpload = document.getElementById('pdf-upload');
    const uploadStatus = document.getElementById('upload-status');
    const voiceControls = document.getElementById('voice-controls');
    const holdToSpeakBtn = document.getElementById('hold-to-speak');
    const micIcon = document.getElementById('mic-icon');
    const recordingIcon = document.getElementById('recording-icon');
    const transcriptDisplay = document.getElementById('transcript-display');
    const transcriptText = transcriptDisplay.querySelector('p');
    const resultsSection = document.getElementById('results-section');
    const noResults = document.getElementById('no-results');
    const qaContainer = document.getElementById('qa-container');
    const summarySection = document.getElementById('summary-section');
    const summaryText = document.getElementById('summary-text');
    const playSummaryBtn = document.getElementById('play-summary');
    const audioPlayer = document.getElementById('audio-player');
    
    // Templates
    const qaTemplate = document.getElementById('qa-template');
    const contextItemTemplate = document.getElementById('context-item-template');
    
    // State
    let currentCollection = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let webSocket = null;
    let currentSummaryAudio = null;
    
    // Initialize WebSocket for real-time transcription
    function initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/audio`;
        
        webSocket = new WebSocket(wsUrl);
        
        webSocket.onopen = () => {
            console.log('WebSocket connection established');
        };
        
        webSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.transcription) {
                transcriptText.textContent = data.transcription;
                transcriptDisplay.classList.remove('hidden');
                
                // If recording has stopped, send the transcription as a query
                if (!isRecording) {
                    processQuery(data.transcription);
                }
            }
        };
        
        webSocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        webSocket.onclose = () => {
            console.log('WebSocket connection closed');
            // Attempt to reconnect after a delay
            setTimeout(initWebSocket, 3000);
        };
    }
    
    // Initialize audio recording
    async function initAudioRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                    
                    // If WebSocket is connected and recording, send audio chunk
                    if (webSocket && webSocket.readyState === WebSocket.OPEN && isRecording) {
                        // Convert to raw audio format for WebSocket
                        event.data.arrayBuffer().then(buffer => {
                            webSocket.send(buffer);
                        });
                    }
                }
            };
            
            mediaRecorder.onstop = async () => {
                // Create audio blob
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                // Create form data
                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.wav');
                
                try {
                    // Send to backend for transcription
                    const response = await fetch('/transcribe-audio', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        transcriptText.textContent = data.text;
                        transcriptDisplay.classList.remove('hidden');
                        
                        // Process the query
                        processQuery(data.text);
                    } else {
                        console.error('Error transcribing audio:', await response.text());
                    }
                } catch (error) {
                    console.error('Error sending audio for transcription:', error);
                }
            };
            
            console.log('Audio recording initialized');
            
        } catch (error) {
            console.error('Error initializing audio recording:', error);
            alert('Could not access microphone. Please check your browser permissions.');
        }
    }
    
    // Upload PDF
    pdfUpload.addEventListener('change', async (event) => {
        if (!event.target.files.length) return;
        
        const file = event.target.files[0];
        if (file.type !== 'application/pdf') {
            alert('Please upload a PDF file.');
            return;
        }
        
        // Display upload status
        uploadStatus.textContent = 'Uploading and processing PDF...';
        uploadStatus.classList.remove('hidden', 'text-red-500');
        uploadStatus.classList.add('text-blue-500');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            // Send to backend
            const response = await fetch('/upload-pdf', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                
                // Store collection name
                currentCollection = data.collection_name;
                
                // Display success
                uploadStatus.textContent = `PDF processed successfully: ${data.num_chunks} chunks created.`;
                uploadStatus.classList.remove('text-blue-500');
                uploadStatus.classList.add('text-green-500');
                
                // Show voice controls
                voiceControls.classList.remove('hidden');
                
                // Show results section
                resultsSection.classList.remove('hidden');
                
                // Show summary
                summarySection.classList.remove('hidden');
                summaryText.textContent = data.summary;
                currentSummaryAudio = null;
                
                // Initialize WebSocket for real-time transcription
                initWebSocket();
                
            } else {
                const error = await response.json();
                uploadStatus.textContent = `Error: ${error.detail}`;
                uploadStatus.classList.remove('text-blue-500');
                uploadStatus.classList.add('text-red-500');
            }
        } catch (error) {
            console.error('Error uploading PDF:', error);
            uploadStatus.textContent = `Error: ${error.message}`;
            uploadStatus.classList.remove('text-blue-500');
            uploadStatus.classList.add('text-red-500');
        }
    });
    
    // Hold to speak functionality
    holdToSpeakBtn.addEventListener('mousedown', () => {
        startRecording();
    });
    
    holdToSpeakBtn.addEventListener('mouseup', () => {
        stopRecording();
    });
    
    holdToSpeakBtn.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startRecording();
    });
    
    holdToSpeakBtn.addEventListener('touchend', () => {
        stopRecording();
    });
    
    // Start recording
    function startRecording() {
        if (!mediaRecorder) {
            initAudioRecording().then(() => {
                startRecordingProcess();
            });
        } else {
            startRecordingProcess();
        }
    }
    
    // Start recording process
    function startRecordingProcess() {
        if (mediaRecorder && mediaRecorder.state === 'inactive') {
            isRecording = true;
            audioChunks = [];
            mediaRecorder.start(100); // Collect data every 100ms
            
            // Update UI
            micIcon.classList.add('hidden');
            recordingIcon.classList.remove('hidden');
            holdToSpeakBtn.parentElement.classList.add('recording');
            
            // Clear previous transcript
            transcriptText.textContent = 'Listening...';
            transcriptDisplay.classList.remove('hidden');
        }
    }
    
    // Stop recording
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            isRecording = false;
            mediaRecorder.stop();
            
            // Update UI
            micIcon.classList.remove('hidden');
            recordingIcon.classList.add('hidden');
            holdToSpeakBtn.parentElement.classList.remove('recording');
            
            transcriptText.textContent = 'Processing...';
        }
    }
    
    // Process query
    async function processQuery(query) {
        if (!currentCollection || !query.trim()) return;
        
        // Show loading state
        noResults.classList.add('hidden');
        qaContainer.classList.remove('hidden');
        
        // Create new QA item
        const qaItem = document.importNode(qaTemplate.content, true).querySelector('.qa-item');
        qaItem.querySelector('.query-text').textContent = query;
        qaItem.querySelector('.answer-text').innerHTML = '<div class="spinner mx-auto"></div>';
        
        // Add to container
        qaContainer.insertBefore(qaItem, qaContainer.firstChild);
        
        try {
            // Send query to backend
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    collection_name: currentCollection,
                    query: query,
                    top_k: 5
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                
                // Update QA item
                qaItem.querySelector('.answer-text').textContent = data.answer;
                
                // Store audio path for playback
                const playButton = qaItem.querySelector('.play-answer');
                playButton.dataset.audioPath = data.audio_path;
                
                // Add context items
                const contextList = qaItem.querySelector('.context-list');
                data.retrieved_chunks.forEach((chunk, index) => {
                    const contextItem = document.importNode(contextItemTemplate.content, true).querySelector('.context-item');
                    contextItem.querySelector('.context-text').textContent = chunk;
                    contextList.appendChild(contextItem);
                });
                
                // Attach play event
                playButton.addEventListener('click', () => {
                    playAudio(data.audio_path);
                });
            } else {
                const error = await response.json();
                qaItem.querySelector('.answer-text').textContent = `Error: ${error.detail}`;
            }
        } catch (error) {
            console.error('Error processing query:', error);
            qaItem.querySelector('.answer-text').textContent = `Error: ${error.message}`;
        }
    }
    
    // Play audio
    function playAudio(audioPath) {
        audioPlayer.src = audioPath;
        audioPlayer.play();
    }
    
    // Play summary
    playSummaryBtn.addEventListener('click', async () => {
        if (currentSummaryAudio) {
            playAudio(currentSummaryAudio);
        } else {
            // Generate summary audio if not already available
            try {
                const response = await fetch('/summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        collection_name: currentCollection,
                        use_full_text: true
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    currentSummaryAudio = data.audio_path;
                    playAudio(data.audio_path);
                } else {
                    const error = await response.json();
                    console.error('Error generating summary audio:', error);
                    alert(`Error: ${error.detail}`);
                }
            } catch (error) {
                console.error('Error:', error);
                alert(`Error: ${error.message}`);
            }
        }
    });
    
    // Initialize
    initAudioRecording();
});