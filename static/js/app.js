// Lore Lantern - Debug Panel JavaScript v6.0 (Session ID + Restart Button)

class DebugPanel {
    constructor() {
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        this.currentStoryId = null;
        // Use relative URL for API - works on any host
        this.apiBaseUrl = '/api';
        this.ws = null;
        this.wsConnected = false;
        this.currentAudio = null;
        this.audioQueue = [];
        this.isPlayingAudio = false;
        this.pingInterval = null;

        // Story data cache
        this.storyData = null;
        this.chaptersData = [];
        this.chapterTTSCache = {};  // { chapterNum: { audio, ssml, duration } }
        this.currentChapterTab = 1;
        this.playbackPhase = 'pre_chapter';

        // Profile data
        this.selectedParentId = null;
        this.selectedChildId = null;
        this.selectedChild = null;  // Full child object with name, birth_year
        this.parentLanguage = 'en';

        // Conversation-first state
        this.conversationTurn = 0;  // Track conversation turns (for exploring mode)
        this.pendingIntent = null;  // Store detected intent for follow-up
        this.pendingActiveStoryId = null;  // Store active story ID from conversation/start
        this.pendingSuggestedAction = null;  // Store suggested action for multi-story handling
        this.preStoryMessages = [];  // Collect user messages before story init for Ch1

        // Chapter Timeline tracking
        this.chapterTimelines = {};  // { chapterNum: { phases: { structure, characters, draft, review, polish }, startTime, ... } }
        this.currentTimelineChapter = 0;

        // Agent duration tracking (track start times to calculate duration in JS)
        this.agentStartTimes = {};  // { "AgentName": timestamp }

        // Streaming dialogue state
        this.streamingMessageBubble = null;  // Current streaming message element
        this.lastStreamingBubble = null;     // Previous streaming message (for audio attachment)

        // Session ID for multi-user identification (tab-specific)
        this.sessionId = this.generateSessionId();

        this.initElements();
        this.initSpeechRecognition();
        this.checkBrowserSupport();
        this.initEventListeners();
        this.checkApiStatus();
        this.loadParents();  // Load existing parents on startup
        this.displaySessionId();  // Show session ID in observatory panel
    }

    // ==================== Session Persistence ====================
    // Persist critical state to localStorage for reconnection after page refresh

    saveSession() {
        const session = {
            parentId: this.selectedParentId,
            childId: this.selectedChildId,
            storyId: this.currentStoryId,
            timestamp: Date.now()
        };
        localStorage.setItem('lorelantern_session', JSON.stringify(session));
        console.log('💾 Session saved:', session);
    }

    loadSession() {
        try {
            const data = localStorage.getItem('lorelantern_session');
            if (!data) return null;

            const session = JSON.parse(data);
            // Check if session is less than 24 hours old
            const MAX_AGE_MS = 24 * 60 * 60 * 1000;  // 24 hours
            if (Date.now() - session.timestamp > MAX_AGE_MS) {
                console.log('🕐 Session expired, clearing...');
                this.clearSession();
                return null;
            }
            return session;
        } catch (e) {
            console.error('Error loading session:', e);
            return null;
        }
    }

    clearSession() {
        localStorage.removeItem('lorelantern_session');
        console.log('🗑️ Session cleared');
    }

    generateSessionId() {
        // Check if we have an existing session ID in sessionStorage (tab-specific)
        let id = sessionStorage.getItem('lorelantern_tab_session_id');
        if (!id) {
            // Generate new ID: 8 chars alphanumeric uppercase
            id = Math.random().toString(36).substring(2, 10).toUpperCase();
            sessionStorage.setItem('lorelantern_tab_session_id', id);
        }
        return id;
    }

    displaySessionId() {
        const sessionIdEl = document.getElementById('session-id-display');
        if (sessionIdEl) {
            sessionIdEl.textContent = this.sessionId;
        }
    }

    async tryRestoreSession() {
        const session = this.loadSession();
        if (!session) return;

        console.log('🔄 Restoring session...', session);

        try {
            // 1. Restore parent selection
            if (session.parentId) {
                // Wait for parents to load
                await new Promise(resolve => setTimeout(resolve, 300));
                this.parentSelect.value = session.parentId;
                await this.handleParentSelect(session.parentId);
            }

            // 2. Restore child selection
            if (session.childId && session.parentId) {
                // Wait for children to load
                await new Promise(resolve => setTimeout(resolve, 500));
                // Find child in grid and select
                const childCard = this.childrenGrid.querySelector(`[data-child-id="${session.childId}"]`);
                if (childCard) {
                    childCard.click();  // Trigger selectChild via click
                }
            }

            // 3. Restore active story
            if (session.storyId && session.childId) {
                await new Promise(resolve => setTimeout(resolve, 500));
                await this.resumeExistingStory(session.storyId);
                console.log('✅ Story restored:', session.storyId);
            }
        } catch (error) {
            console.error('Failed to restore session:', error);
            this.clearSession();
        }
    }

    initElements() {
        // Profile elements
        this.parentSelect = document.getElementById('parent-select');
        this.createParentBtn = document.getElementById('create-parent-btn');
        this.createParentForm = document.getElementById('create-parent-form');
        this.saveParentBtn = document.getElementById('save-parent-btn');
        this.cancelParentBtn = document.getElementById('cancel-parent-btn');
        this.parentLanguageBadge = document.getElementById('parent-language-badge');

        this.childrenGrid = document.getElementById('children-grid');
        this.createChildBtn = document.getElementById('create-child-btn');
        this.createChildForm = document.getElementById('create-child-form');
        this.saveChildBtn = document.getElementById('save-child-btn');
        this.cancelChildBtn = document.getElementById('cancel-child-btn');

        this.activeProfileSummary = document.getElementById('active-profile-summary');
        this.storyCreationPanel = document.getElementById('story-creation-panel');

        // Story creation
        this.storyInput = document.getElementById('story-input');
        this.sendButton = document.getElementById('send-button');
        this.micButton = document.getElementById('mic-button');
        this.micStatus = document.getElementById('mic-status');

        // Story details
        this.storyDetailsPanel = document.getElementById('story-details-panel');
        this.playbackPhaseBadge = document.getElementById('playback-phase');
        this.refreshStoryBtn = document.getElementById('refresh-story-btn');

        // Chapters
        this.chaptersPanel = document.getElementById('chapters-panel');
        this.chapterTabs = document.getElementById('chapter-tabs');
        this.chapterContentArea = document.getElementById('chapter-content-area');
        this.generateAllTTSBtn = document.getElementById('generate-all-tts-btn');
        this.refineNorwegianBtn = document.getElementById('refine-norwegian-btn');
        this.compareRefinementBtn = document.getElementById('compare-refinement-btn');

        // Comparison Modal
        this.comparisonModal = document.getElementById('refinement-comparison-modal');
        this.closeComparisonModalBtn = document.getElementById('close-comparison-modal');
        this.comparisonOriginal = document.getElementById('comparison-original');
        this.comparisonRefined = document.getElementById('comparison-refined');
        this.statsOriginalWords = document.getElementById('stats-original-words');
        this.statsRefinedWords = document.getElementById('stats-refined-words');
        this.statsDiff = document.getElementById('stats-diff');
        this.statsStatus = document.getElementById('stats-status');
        this.toggleDiffViewBtn = document.getElementById('toggle-diff-view');

        // Store last refinement data for comparison
        this.lastRefinementData = null;

        // Chat
        this.chatHistory = document.getElementById('chat-history');
        this.chatInputContainer = document.getElementById('chat-input-container');
        this.chatInput = document.getElementById('chat-input');
        this.sendChatBtn = document.getElementById('send-chat-btn');
        this.discussionIndicator = document.getElementById('discussion-indicator');

        // Debug
        this.stopSpeechBtn = document.getElementById('stop-speech-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.testApiBtn = document.getElementById('test-api-btn');
        this.wsStatus = document.getElementById('ws-status');
        this.rawJson = document.getElementById('raw-json');

        // Restart session button (in profile panel)
        this.restartSessionBtn = document.getElementById('restart-session-btn');
    }

    initSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return;

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateMicStatus('Listening...', 'listening');
        };

        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.storyInput.value = transcript;
            this.updateMicStatus('Got it!', 'idle');
        };

        this.recognition.onerror = (event) => {
            this.updateMicStatus(`Error: ${event.error}`, 'idle');
            this.isListening = false;
        };

        this.recognition.onend = () => {
            this.isListening = false;
        };
    }

    initEventListeners() {
        // Profile management
        this.parentSelect?.addEventListener('change', (e) => this.handleParentSelect(e.target.value));
        this.createParentBtn?.addEventListener('click', () => this.showCreateParentForm());
        this.saveParentBtn?.addEventListener('click', () => this.createParent());
        this.cancelParentBtn?.addEventListener('click', () => this.hideCreateParentForm());
        this.createChildBtn?.addEventListener('click', () => this.showCreateChildForm());
        this.saveChildBtn?.addEventListener('click', () => this.createChild());
        this.cancelChildBtn?.addEventListener('click', () => this.hideCreateChildForm());

        // Story creation
        this.sendButton.addEventListener('click', () => this.handleCreateStory());
        this.storyInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleCreateStory();
        });

        // Voice input
        this.micButton.addEventListener('click', () => {
            if (this.recognition) {
                if (this.isListening) {
                    this.recognition.stop();
                } else {
                    this.recognition.start();
                }
            }
        });

        // Chat
        this.sendChatBtn.addEventListener('click', () => this.handleChatInput());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleChatInput();
        });

        // Refresh story
        this.refreshStoryBtn?.addEventListener('click', () => {
            if (this.currentStoryId) this.fetchStoryDetails(this.currentStoryId);
        });

        // Generate all TTS
        this.generateAllTTSBtn?.addEventListener('click', () => this.generateAllTTS());

        // Refine Norwegian (Borealis)
        this.refineNorwegianBtn?.addEventListener('click', () => this.refineNorwegian());

        // Compare Refinement
        this.compareRefinementBtn?.addEventListener('click', () => this.showComparisonModal());
        this.closeComparisonModalBtn?.addEventListener('click', () => this.hideComparisonModal());
        this.comparisonModal?.querySelector('.modal-overlay')?.addEventListener('click', () => this.hideComparisonModal());
        this.toggleDiffViewBtn?.addEventListener('click', () => this.toggleDiffView());

        // Controls
        this.stopSpeechBtn.addEventListener('click', () => this.stopAudio());
        this.clearBtn.addEventListener('click', () => this.clearAll());
        this.testApiBtn.addEventListener('click', () => this.checkApiStatus());

        // Restart Session button (in profile panel, visible after profile selected)
        this.restartSessionBtn?.addEventListener('click', () => this.clearAll());

        // Clear Event Log button
        const clearEventLogBtn = document.getElementById('clear-eventlog-btn');
        clearEventLogBtn?.addEventListener('click', () => this.clearEventLog());
    }

    // ==================== Profile Management ====================

    async loadParents() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/parents`);
            if (!response.ok) return;

            const data = await response.json();
            this.populateParentDropdown(data.parents || []);
        } catch (error) {
            console.log('No existing parents found');
        }
    }

    populateParentDropdown(parents) {
        if (!this.parentSelect) return;

        // Keep the default option
        this.parentSelect.innerHTML = '<option value="">Select or create parent...</option>';

        parents.forEach(parent => {
            const option = document.createElement('option');
            option.value = parent.parent_id;
            option.textContent = `${parent.display_name || parent.parent_id} (${this.getLanguageFlag(parent.language)})`;
            this.parentSelect.appendChild(option);
        });
    }

    getLanguageFlag(lang) {
        const flags = { en: '🇬🇧', no: '🇳🇴', es: '🇪🇸' };
        return flags[lang] || '🌍';
    }

    async handleParentSelect(parentId) {
        if (!parentId) {
            this.selectedParentId = null;
            this.parentLanguage = 'en';
            this.parentLanguageBadge.textContent = '🌍 --';
            this.createChildBtn.disabled = true;
            this.childrenGrid.innerHTML = '<p class="placeholder">Select a household first...</p>';
            this.hideStoryCreation();
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/parents/${parentId}`);
            if (!response.ok) throw new Error('Parent not found');

            const data = await response.json();
            const parent = data.account;  // API returns {success, account: {...}}
            this.selectedParentId = parentId;
            this.parentLanguage = parent.language || 'en';
            this.parentLanguageBadge.textContent = `${this.getLanguageFlag(parent.language)} ${parent.language.toUpperCase()}`;
            this.createChildBtn.disabled = false;

            // Load children for this parent
            await this.loadChildren(parentId);

            // Save session for reconnection
            this.saveSession();
        } catch (error) {
            console.error('Error loading parent:', error);
            this.addSystemMessage(`Error: ${error.message}`, 'error');
        }
    }

    async loadChildren(parentId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/parents/${parentId}/children`);
            if (!response.ok) throw new Error('Could not load children');

            const data = await response.json();
            this.displayChildren(data.children || []);
        } catch (error) {
            this.childrenGrid.innerHTML = '<p class="placeholder">No children yet. Create one!</p>';
        }
    }

    displayChildren(children) {
        this.childrenGrid.innerHTML = '';

        if (!children.length) {
            this.childrenGrid.innerHTML = '<p class="placeholder">No children yet. Create one!</p>';
            return;
        }

        children.forEach(child => {
            const card = document.createElement('div');
            card.className = 'child-profile-card';
            card.dataset.childId = child.child_id;

            // API returns current_age, but also support birth_year fallback
            const age = child.current_age || (child.birth_year ? new Date().getFullYear() - child.birth_year : 8);
            const avatar = age <= 5 ? '👶' : age <= 10 ? '👧' : '🧒';

            card.innerHTML = `
                <span class="child-avatar">${avatar}</span>
                <div class="child-info">
                    <span class="child-name">${child.name}</span>
                    <span class="child-age">${age} years old</span>
                </div>
                <span class="story-count">${child.story_count || 0} stories</span>
            `;

            card.addEventListener('click', () => this.selectChild(child, age, avatar));
            this.childrenGrid.appendChild(card);
        });
    }

    selectChild(child, age, avatar) {
        // Update selection state
        document.querySelectorAll('.child-profile-card').forEach(c => c.classList.remove('selected'));
        const selectedCard = this.childrenGrid.querySelector(`[data-child-id="${child.child_id}"]`);
        if (selectedCard) selectedCard.classList.add('selected');

        this.selectedChildId = child.child_id;
        this.selectedChild = child;

        // Update active profile summary
        document.getElementById('active-avatar').textContent = avatar;
        document.getElementById('active-name').textContent = child.name;
        document.getElementById('active-age').textContent = `${age} years old`;

        this.activeProfileSummary.classList.remove('hidden');
        this.storyCreationPanel.classList.remove('hidden');

        this.addSystemMessage(`Selected: ${child.name} (${age} years old)`);

        // Save session for reconnection
        this.saveSession();
    }

    showCreateParentForm() {
        this.createParentForm.classList.remove('hidden');
    }

    hideCreateParentForm() {
        this.createParentForm.classList.add('hidden');
        document.getElementById('new-parent-name').value = '';
    }

    async createParent() {
        const displayName = document.getElementById('new-parent-name').value.trim();
        const language = document.getElementById('new-parent-language').value;

        if (!displayName) {
            alert('Please enter a family name');
            return;
        }

        try {
            // Let backend auto-generate UUID for parent_id
            const response = await fetch(`${this.apiBaseUrl}/parents`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    display_name: displayName,
                    language: language
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Failed to create parent');
            }

            const data = await response.json();
            const parent = data.account;  // API returns {success, parent_id, message, account}
            this.addSystemMessage(`Created family: ${parent.display_name}`);

            // Reload parents and select the new one
            await this.loadParents();
            this.parentSelect.value = parent.parent_id;
            await this.handleParentSelect(parent.parent_id);

            this.hideCreateParentForm();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    showCreateChildForm() {
        this.createChildForm.classList.remove('hidden');
    }

    hideCreateChildForm() {
        this.createChildForm.classList.add('hidden');
        // Support both old and new element IDs
        const nameEl = document.getElementById('new-user-name') || document.getElementById('new-child-name');
        const ageEl = document.getElementById('new-user-age') || document.getElementById('new-child-age');
        if (nameEl) nameEl.value = '';
        if (ageEl) ageEl.value = '';
    }

    async createChild() {
        console.log('createChild called');  // Debug
        const nameEl = document.getElementById('new-user-name') || document.getElementById('new-child-name');
        const ageEl = document.getElementById('new-user-age') || document.getElementById('new-child-age');

        const name = nameEl?.value?.trim();
        const age = parseInt(ageEl?.value);

        console.log('Name:', name, 'Age:', age);  // Debug

        if (!name || !age || isNaN(age)) {
            alert('Please enter name and age');
            return;
        }

        if (!this.selectedParentId) {
            alert('Please select a household first');
            return;
        }

        // Calculate birth year from age
        const currentYear = new Date().getFullYear();
        const birthYear = currentYear - age;

        try {
            const response = await fetch(`${this.apiBaseUrl}/parents/${this.selectedParentId}/children`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    birth_year: birthYear
                })
            });

            if (!response.ok) {
                const err = await response.json();
                // Handle Pydantic validation errors (detail is an array)
                let errorMsg = 'Failed to create child';
                if (err.detail) {
                    if (Array.isArray(err.detail)) {
                        errorMsg = err.detail.map(e => e.msg || e.message || JSON.stringify(e)).join(', ');
                    } else if (typeof err.detail === 'string') {
                        errorMsg = err.detail;
                    }
                }
                throw new Error(errorMsg);
            }

            const data = await response.json();
            const child = data.profile;  // API returns {success, child_id, message, profile}
            this.addSystemMessage(`Created user: ${child.name} (${age} years old)`);

            // Reload children
            await this.loadChildren(this.selectedParentId);
            this.hideCreateChildForm();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }

    hideStoryCreation() {
        this.selectedChildId = null;
        this.selectedChild = null;
        this.activeProfileSummary?.classList.add('hidden');
        this.storyCreationPanel?.classList.add('hidden');
    }

    // ==================== Story Creation ====================

    async handleCreateStory() {
        const text = this.storyInput.value.trim();
        if (!text) {
            alert('Please enter a story request');
            return;
        }

        if (!this.selectedChildId) {
            alert('Please select a child profile first');
            return;
        }

        this.sendButton.textContent = 'Thinking...';
        this.sendButton.disabled = true;

        try {
            // ===== CONVERSATION-FIRST: Call /conversation/start first =====
            const conversationResponse = await fetch(`${this.apiBaseUrl}/conversation/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    child_id: this.selectedChildId,
                    language: this.parentLanguage
                })
            });

            if (!conversationResponse.ok) throw new Error(`API error: ${conversationResponse.status}`);

            const convData = await conversationResponse.json();

            // Track pre-story user messages for Chapter 1 context
            this.preStoryMessages.push(text);

            // Add user message and CompanionAgent greeting to chat
            this.addChatMessage(text, 'user');
            this.addChatMessage(convData.dialogue, 'narrator', convData.audio);

            // Store intent state for follow-up
            this.pendingIntent = convData.intent;
            this.pendingActiveStoryId = convData.active_story_id;
            this.pendingSuggestedAction = convData.suggested_action;
            this.conversationTurn = 1;

            console.log(`🎯 Intent detected: ${convData.intent}, suggested_action: ${convData.suggested_action}`);

            // Route based on intent
            if (convData.intent === 'continue' && convData.suggested_action === 'resume_story' && convData.active_story_id) {
                // Child wants to continue - resume existing story
                await this.resumeExistingStory(convData.active_story_id);
            } else if (convData.intent === 'new_story' || convData.suggested_action === 'init_story') {
                // Child wants new story - proceed to story generation
                await this.initializeNewStory(text);
            } else {
                // Exploring, greeting, or ask_preference (multiple stories) - show chat panel for continued conversation
                this.chatInputContainer.classList.remove('hidden');
                this.updateMicStatus('Continue chatting!', 'idle');
            }

            // Clear input
            this.storyInput.value = '';

        } catch (error) {
            console.error('Error starting conversation:', error);
            alert(`Error: ${error.message}`);
        } finally {
            this.sendButton.textContent = 'Start';
            this.sendButton.disabled = false;
        }
    }

    async resumeExistingStory(storyId) {
        /**
         * Resume an existing story via the continue endpoint.
         */
        try {
            this.updateMicStatus('Resuming story...', 'idle');

            const response = await fetch(`${this.apiBaseUrl}/children/${this.selectedChildId}/stories/${storyId}/continue`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) throw new Error(`Continue API error: ${response.status}`);

            const data = await response.json();
            this.currentStoryId = storyId;

            // Save session for reconnection
            this.saveSession();

            // Show story panels
            this.storyDetailsPanel.classList.remove('hidden');
            this.chaptersPanel.classList.remove('hidden');
            this.chatInputContainer.classList.remove('hidden');

            // Connect WebSocket
            this.connectWebSocket(storyId);

            // Fetch story details
            await this.fetchStoryDetails(storyId);

            this.updateMicStatus('Story resumed!', 'idle');

            // Reset conversation state
            this.conversationTurn = 0;
            this.pendingIntent = null;
            this.pendingSuggestedAction = null;

        } catch (error) {
            console.error('Error resuming story:', error);
            this.addSystemMessage(`Error resuming story: ${error.message}`, 'error');
        }
    }

    async initializeNewStory(prompt) {
        /**
         * Initialize a new story via /conversation/init.
         */
        try {
            this.updateMicStatus('Creating story...', 'idle');

            // Collect any pre-story messages to pass for Chapter 1 context
            const preStoryMsgs = this.preStoryMessages.length > 0 ? [...this.preStoryMessages] : null;

            const response = await fetch(`${this.apiBaseUrl}/conversation/init`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    child_id: this.selectedChildId,
                    language: this.parentLanguage,
                    pre_story_messages: preStoryMsgs
                    // chapters generated one-at-a-time via prefetch
                })
            });

            if (!response.ok) {
                // Parse error response for user-friendly message
                const errorData = await response.json().catch(() => null);
                if (errorData?.detail?.error === 'content_too_long') {
                    const { actual_length, max_length, message } = errorData.detail;
                    throw new Error(message || `Prompt too long (${actual_length.toLocaleString()} chars, max ${max_length.toLocaleString()})`);
                }
                throw new Error(errorData?.detail?.message || errorData?.detail || `Init API error: ${response.status}`);
            }

            const data = await response.json();
            this.currentStoryId = data.story_id;

            // Save session for reconnection
            this.saveSession();

            // Add welcome message
            this.addChatMessage(data.welcome_message, 'narrator', data.audio);

            // Show panels
            this.storyDetailsPanel.classList.remove('hidden');
            this.chaptersPanel.classList.remove('hidden');
            this.chatInputContainer.classList.remove('hidden');

            // Connect WebSocket
            this.connectWebSocket(data.story_id);

            // Fetch full story details
            setTimeout(() => this.fetchStoryDetails(data.story_id), 2000);

            this.updateMicStatus('Story created!', 'idle');

            // Reset conversation state
            this.conversationTurn = 0;
            this.pendingIntent = null;
            this.pendingSuggestedAction = null;
            this.preStoryMessages = [];  // Clear after story init

        } catch (error) {
            console.error('Error initializing story:', error);
            this.addSystemMessage(`Error creating story: ${error.message}`, 'error');
        }
    }

    // ==================== Story Data Fetching ====================

    async fetchStoryDetails(storyId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/stories/${storyId}`);
            if (!response.ok) throw new Error(`API error: ${response.status}`);

            const data = await response.json();
            this.storyData = data.story;

            // Display all data in new Story Info panel
            this.displayStoryInfo(data.story);
            this.displayCharacters(data.story.characters || []);
            this.displayEducationalGoals(data.story.structure?.educational_goals || []);
            this.displayPlotThreads(data.story.structure?.plot_elements || []);
            this.displayChapterOutlines(data.story.structure?.chapters || [], data.story.chapters || []);
            this.displayChapters(data.story.chapters || []);

            // Show/hide Norwegian refinement button based on story language
            const isNorwegian = data.story.preferences?.language === 'no';
            if (this.refineNorwegianBtn) {
                this.refineNorwegianBtn.classList.toggle('hidden', !isNorwegian);
            }

            // Update raw JSON
            const rawEl = document.getElementById('info-raw-json');
            if (rawEl) rawEl.textContent = JSON.stringify(data.story, null, 2);

            console.log('Story data loaded:', data.story);

        } catch (error) {
            console.error('Error fetching story:', error);
            this.addSystemMessage(`Error loading story: ${error.message}`, 'error');
        }
    }

    displayStoryInfo(story) {
        // Title
        const titleEl = document.getElementById('info-title');
        if (titleEl) titleEl.textContent = story.structure?.title || 'Untitled';

        // Status badge
        const statusEl = document.getElementById('info-status');
        if (statusEl) {
            statusEl.textContent = story.status || 'unknown';
            statusEl.className = `status-badge status-${story.status}`;
        }

        // Meta tags
        const themeEl = document.getElementById('info-theme');
        if (themeEl) themeEl.textContent = `🎭 ${story.structure?.theme || 'No theme'}`;

        const methodEl = document.getElementById('info-method');
        if (methodEl) {
            let method = story.structure?.narrative_method || 'LINEAR_SINGLE';
            // Handle if narrative_method is an object with a name property
            if (typeof method === 'object' && method !== null) {
                method = method.name || method.type || 'LINEAR';
            }
            methodEl.textContent = `📖 ${String(method).replace(/_/g, ' ')}`;
        }

        const readingTimeEl = document.getElementById('info-reading-time');
        if (readingTimeEl) {
            const mins = story.structure?.estimated_reading_time_minutes || '?';
            readingTimeEl.textContent = `⏱️ ${mins} min`;
        }

        // Progress bar
        const totalChapters = story.structure?.chapters?.length || 1;
        const completedChapters = (story.chapters || []).filter(ch => ch.content).length;
        const progress = Math.round((completedChapters / totalChapters) * 100);

        const progressBar = document.getElementById('info-progress');
        if (progressBar) progressBar.style.width = `${progress}%`;

        const progressLabel = document.getElementById('info-progress-label');
        if (progressLabel) progressLabel.textContent = `${completedChapters}/${totalChapters} chapters (${progress}%)`;
    }

    displayEducationalGoals(goals) {
        const container = document.getElementById('info-goals');
        if (!container) return;

        if (!goals.length) {
            container.innerHTML = '<p class="placeholder">No learning goals defined</p>';
            return;
        }

        container.innerHTML = goals.map(goal => `
            <div class="goal-item">
                <div class="goal-concept">${goal.concept}</div>
                <div class="goal-description">${goal.description}</div>
            </div>
        `).join('');
    }

    displayPlotThreads(threads) {
        const container = document.getElementById('info-plot-threads');
        if (!container) return;

        if (!threads.length) {
            container.innerHTML = '<p class="placeholder">No plot threads tracked</p>';
            return;
        }

        container.innerHTML = threads.map(thread => {
            const statusClass = thread.resolved ? 'resolved' : (thread.setup_chapter ? 'active' : 'setup');
            const statusText = thread.resolved ? `✓ Ch${thread.payoff_chapter}` :
                               thread.setup_chapter ? `Setup Ch${thread.setup_chapter}` : 'Pending';
            return `
                <div class="plot-thread ${statusClass}">
                    <span class="thread-name">${thread.element || thread.name}</span>
                    <span class="thread-status">${statusText}</span>
                </div>
            `;
        }).join('');
    }

    displayChapterOutlines(outlines, writtenChapters) {
        const container = document.getElementById('info-chapters');
        if (!container) return;

        if (!outlines.length) {
            container.innerHTML = '<p class="placeholder">No chapter outlines</p>';
            return;
        }

        const writtenMap = new Map(writtenChapters.map(ch => [ch.chapter_number || ch.number, ch]));

        container.innerHTML = outlines.map(outline => {
            const written = writtenMap.get(outline.number);
            const statusIcon = written?.content ? '✅' : (written ? '⏳' : '○');
            return `
                <div class="chapter-outline-item" onclick="this.classList.toggle('expanded')">
                    <div class="chapter-outline-header">
                        <span class="chapter-num">Ch${outline.number}</span>
                        <span class="chapter-title">${outline.title}</span>
                        <span class="chapter-status-icon">${statusIcon}</span>
                    </div>
                    <div class="chapter-synopsis">${outline.synopsis || 'No synopsis'}</div>
                </div>
            `;
        }).join('');
    }

    displayCharacters(characters) {
        const grid = document.getElementById('characters-grid');
        if (!grid) return;
        grid.innerHTML = '';

        if (!characters.length) {
            grid.innerHTML = '<p class="placeholder">No characters yet...</p>';
            return;
        }

        characters.forEach(char => {
            const importance = char.importance || 'supporting';
            const traits = (char.personality_traits || []).slice(0, 2).join(', ');
            const card = document.createElement('div');
            card.className = `character-card ${importance}`;
            card.title = char.arc || char.backstory || '';
            card.innerHTML = `
                <div class="char-name">${char.name}</div>
                <div class="char-role">${char.role}</div>
                ${traits ? `<div class="char-traits">${traits}</div>` : ''}
            `;
            grid.appendChild(card);
        });
    }

    displayChapters(chapters) {
        console.log('📖 displayChapters called with:', chapters);

        if (!chapters.length) {
            this.chaptersData = [];
            this.chapterContentArea.innerHTML = '<div class="chapter-placeholder"><p>No chapters yet...</p></div>';
            return;
        }

        // Normalize chapter data - ensure each chapter has a chapter_number
        this.chaptersData = chapters.map((ch, idx) => ({
            ...ch,
            chapter_number: ch.chapter_number ?? ch.chapterNumber ?? (idx + 1)
        }));

        this.chapterTabs.innerHTML = '';

        // Create tabs
        this.chaptersData.forEach((ch, idx) => {
            console.log(`📖 Chapter ${idx}:`, ch, 'chapter_number:', ch.chapter_number);
            const tab = document.createElement('button');
            tab.className = `chapter-tab ${idx === 0 ? 'active' : ''}`;
            tab.dataset.chapter = ch.chapter_number;
            tab.innerHTML = `
                Ch ${ch.chapter_number}
                <span class="tab-status ${ch.status || 'pending'}">${this.getStatusEmoji(ch.status)}</span>
            `;
            tab.addEventListener('click', () => this.selectChapter(ch.chapter_number));
            this.chapterTabs.appendChild(tab);
        });

        // Preserve current chapter selection, or default to first chapter
        const chapterToSelect = this.currentChapterTab &&
            this.chaptersData.some(ch => ch.chapter_number === this.currentChapterTab)
            ? this.currentChapterTab
            : this.chaptersData[0].chapter_number;
        console.log('📖 Selecting chapter:', chapterToSelect, '(currentChapterTab was:', this.currentChapterTab, ')');
        this.selectChapter(chapterToSelect);
    }

    getStatusEmoji(status) {
        switch (status) {
            case 'ready': return '✅';
            case 'generating': return '⏳';
            case 'error': return '❌';
            default: return '⏸️';
        }
    }

    selectChapter(chapterNum) {
        console.log('📖 selectChapter called with:', chapterNum, 'type:', typeof chapterNum);
        console.log('📖 chaptersData:', this.chaptersData);

        // Ensure chapterNum is a number for comparison
        const targetChapter = typeof chapterNum === 'string' ? parseInt(chapterNum) : chapterNum;
        this.currentChapterTab = targetChapter;

        // Update tab active state
        document.querySelectorAll('.chapter-tab').forEach(tab => {
            const tabChapter = parseInt(tab.dataset.chapter);
            tab.classList.toggle('active', tabChapter === targetChapter);
        });

        // Find chapter data (chaptersData is already normalized with chapter_number)
        const chapter = this.chaptersData.find(ch => ch.chapter_number === targetChapter);
        console.log('📖 Found chapter:', chapter);
        if (!chapter) {
            this.chapterContentArea.innerHTML = '<div class="chapter-placeholder"><p>Chapter not found</p></div>';
            return;
        }

        // Check TTS cache - also check for blob URL in chapter data
        let ttsData = this.chapterTTSCache[chapterNum];

        // If chapter has a cached blob URL but no local cache, create cache entry
        if (!ttsData && chapter.audio_blob_url) {
            ttsData = {
                audioUrl: chapter.audio_blob_url,
                source: 'blob_cache'
            };
            this.chapterTTSCache[chapterNum] = ttsData;
            console.log('📖 Using cached audio from Azure Blob:', chapter.audio_blob_url.substring(0, 50));
        }

        this.chapterContentArea.innerHTML = `
            <div class="chapter-view">
                <div class="chapter-header">
                    <h3>Chapter ${chapter.chapter_number}: ${chapter.title || 'Untitled'}</h3>
                    <div class="chapter-meta">
                        <span>📝 ${chapter.word_count || '?'} words</span>
                        <span>⏱️ ${chapter.reading_time_minutes || '?'} min read</span>
                        <span class="status-badge status-${chapter.status}">${chapter.status || 'unknown'}</span>
                    </div>
                </div>

                ${chapter.synopsis ? `<div class="chapter-synopsis"><strong>Synopsis:</strong> ${chapter.synopsis}</div>` : ''}

                <div class="chapter-prose">
                    ${chapter.content || '<em>No content yet...</em>'}
                </div>

                ${chapter.vocabulary_words?.length ? `
                    <div class="chapter-vocab">
                        <strong>Vocabulary:</strong>
                        ${chapter.vocabulary_words.map(v => `<span class="vocab-word" title="${v.definition}">${v.word}</span>`).join(', ')}
                    </div>
                ` : ''}

                <div class="tts-controls">
                    <!-- Step 1: Voice Direction (SSML/Audio Tags) -->
                    <div class="tts-section">
                        <div class="tts-section-header">
                            <span class="section-label">1. Voice Direction</span>
                            <span class="section-status ${chapter.tts_content ? 'ready' : 'pending'}">
                                ${chapter.tts_content ? `✓ Ready (${chapter.tts_content.length} chars)` : '⏳ Not generated'}
                            </span>
                        </div>
                        <div class="tts-section-actions">
                            <button class="action-btn"
                                    onclick="debugPanel.regenerateSSML(${chapterNum})"
                                    title="Run VoiceDirectorAgent to generate audio tags for expressive narration">
                                🔄 Regenerate Voice Tags
                            </button>
                            <span class="action-hint">VoiceDirectorAgent → adds [emotion] tags</span>
                        </div>
                    </div>

                    <!-- Step 2: TTS Audio Generation -->
                    <div class="tts-section">
                        <div class="tts-section-header">
                            <span class="section-label">2. TTS Audio</span>
                            <span class="section-status ${chapter.audio_blob_url ? 'cached' : (ttsData ? 'ready' : 'pending')}" id="audio-status-${chapterNum}">
                                ${chapter.audio_blob_url
                                    ? '☁️ Cached in Azure Blob'
                                    : (ttsData
                                        ? '✓ Generated (in memory)'
                                        : '⏳ Not generated')}
                            </span>
                        </div>
                        <div class="tts-section-actions">
                            <button class="action-btn primary"
                                    onclick="debugPanel.generateChapterTTS(${chapterNum})"
                                    ${chapter.tts_content ? '' : 'disabled'}
                                    title="${chapter.tts_content ? 'Generate MP3 audio from voice tags' : 'Need voice tags first'}">
                                🔊 Generate Audio
                            </button>
                            <button class="action-btn"
                                    onclick="debugPanel.regenerateTTS(${chapterNum})"
                                    ${chapter.tts_content ? '' : 'disabled'}
                                    title="Force regenerate audio (ignores Azure cache)">
                                🔄 Force Regenerate
                            </button>
                            <span class="action-hint">ElevenLabs/Speechify → MP3</span>
                        </div>
                    </div>

                    <!-- Step 3: Audio Player -->
                    <div class="tts-section">
                        <div class="tts-section-header">
                            <span class="section-label">3. Playback</span>
                        </div>
                        <div class="audio-player-box" id="audio-player-${chapterNum}">
                            <div class="player-buttons">
                                <button class="player-btn"
                                        onclick="debugPanel.playChapterTTS(${chapterNum})"
                                        ${ttsData || chapter.audio_blob_url ? '' : 'disabled'}
                                        id="play-btn-${chapterNum}"
                                        title="Play audio">
                                    ▶ Play
                                </button>
                                <button class="player-btn"
                                        onclick="debugPanel.pauseChapterTTS(${chapterNum})"
                                        disabled
                                        id="pause-btn-${chapterNum}"
                                        title="Pause audio">
                                    ⏸ Pause
                                </button>
                                <button class="player-btn"
                                        onclick="debugPanel.stopChapterTTS(${chapterNum})"
                                        disabled
                                        id="stop-btn-${chapterNum}"
                                        title="Stop and reset">
                                    ⏹ Stop
                                </button>
                            </div>
                            <div class="progress-row">
                                <input type="range" class="progress-bar"
                                       id="progress-${chapterNum}"
                                       min="0" max="100" value="0"
                                       oninput="debugPanel.seekChapterTTS(${chapterNum}, this.value)">
                                <span class="time-display" id="time-${chapterNum}">0:00 / 0:00</span>
                            </div>
                            <div class="player-actions">
                                <button class="action-btn small"
                                        onclick="debugPanel.downloadChapterTTS(${chapterNum})"
                                        ${ttsData || chapter.audio_blob_url ? '' : 'disabled'}>
                                    📥 Download MP3
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                ${chapter.tts_content ? `
                    <details class="ssml-viewer">
                        <summary>📄 SSML Output (from VoiceDirectorAgent)</summary>
                        <pre>${this.escapeHtml(chapter.tts_content)}</pre>
                    </details>
                ` : ''}

                ${ttsData?.ssml ? `
                    <details class="ssml-viewer">
                        <summary>📄 Generated Audio SSML</summary>
                        <pre>${this.escapeHtml(ttsData.ssml)}</pre>
                    </details>
                ` : ''}
            </div>
        `;

        // Show/hide compare button based on whether chapter has refinement data
        if (this.compareRefinementBtn) {
            const hasRefinementData = chapter.language_refined && chapter.pre_refinement_content;
            console.log('🔍 Refinement check:', {
                chapterNum: chapter.chapter_number,
                language_refined: chapter.language_refined,
                has_pre_refinement: !!chapter.pre_refinement_content,
                hasRefinementData
            });
            this.compareRefinementBtn.classList.toggle('hidden', !hasRefinementData);
        }
    }

    // ==================== TTS Controls ====================

    async generateChapterTTS(chapterNum) {
        // Fallback to currentChapterTab if chapterNum is undefined
        const chapter = chapterNum ?? this.currentChapterTab;
        if (!chapter) {
            this.addSystemMessage('No chapter selected', 'error');
            return;
        }

        const statusEl = document.getElementById(`tts-status-${chapter}`);
        if (statusEl) statusEl.textContent = '⏳ Generating audio from SSML...';

        try {
            // Call the NEW /generate-audio endpoint (converts SSML to actual audio)
            const response = await fetch(
                `${this.apiBaseUrl}/stories/${this.currentStoryId}/chapters/${chapter}/generate-audio`,
                { method: 'POST' }
            );

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `API error: ${response.status}`);
            }

            const data = await response.json();

            // Cache the audio data for playback
            // API returns: audio_url (blob SAS URL) or fallback_audio_base64 (base64)
            this.chapterTTSCache[chapter] = {
                audio: data.fallback_audio_base64,  // Base64 audio (fallback)
                audioUrl: data.audio_url,           // Blob storage SAS URL
                duration: data.duration_estimate,
                provider: data.provider,
                source: data.source
            };

            // Refresh display
            this.selectChapter(chapter);

            this.addSystemMessage(`🔊 Audio generated for Chapter ${chapter} (~${data.duration_estimate}s)`);

        } catch (error) {
            console.error('TTS generation error:', error);
            if (statusEl) statusEl.textContent = `❌ Error: ${error.message}`;
        }
    }

    playChapterTTS(chapterNum) {
        const chapter = chapterNum ?? this.currentChapterTab;
        const ttsData = this.chapterTTSCache[chapter];

        // Also check chapter data for audio_blob_url
        const chapterData = this.chaptersData.find(ch => ch.chapter_number === chapter);
        const blobUrl = ttsData?.audioUrl || chapterData?.audio_blob_url;

        let audioSrc = null;

        // Try blob storage URL first (more efficient), then base64 fallback
        if (blobUrl) {
            audioSrc = blobUrl;
            this.addSystemMessage(`▶️ Playing Chapter ${chapter} (from cloud)...`);
        } else if (ttsData?.audio) {
            // Convert base64 to blob URL
            const blob = this.base64ToBlob(ttsData.audio, 'audio/mp3');
            audioSrc = URL.createObjectURL(blob);
            this.addSystemMessage(`▶️ Playing Chapter ${chapter} (local)...`);
        } else {
            // Try to fetch from Azure first
            this.fetchCachedAudio(chapter);
            return;
        }

        // Create audio element with player controls
        this.setupChapterAudioPlayer(chapter, audioSrc);
    }

    setupChapterAudioPlayer(chapterNum, audioSrc) {
        // Stop any existing audio for this chapter
        if (this.chapterAudioElements?.[chapterNum]) {
            this.chapterAudioElements[chapterNum].pause();
        }

        // Initialize audio elements map if needed
        if (!this.chapterAudioElements) {
            this.chapterAudioElements = {};
        }

        const audio = new Audio(audioSrc);
        this.chapterAudioElements[chapterNum] = audio;

        // Get control elements
        const playBtn = document.getElementById(`play-btn-${chapterNum}`);
        const pauseBtn = document.getElementById(`pause-btn-${chapterNum}`);
        const stopBtn = document.getElementById(`stop-btn-${chapterNum}`);
        const progressBar = document.getElementById(`progress-${chapterNum}`);
        const timeDisplay = document.getElementById(`time-${chapterNum}`);

        // Enable/disable buttons
        if (playBtn) playBtn.disabled = true;
        if (pauseBtn) pauseBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = false;

        // Update progress bar as audio plays
        audio.addEventListener('timeupdate', () => {
            if (progressBar && audio.duration) {
                progressBar.value = (audio.currentTime / audio.duration) * 100;
            }
            if (timeDisplay) {
                timeDisplay.textContent = `${this.formatTime(audio.currentTime)} / ${this.formatTime(audio.duration || 0)}`;
            }
        });

        // Reset when audio ends
        audio.addEventListener('ended', () => {
            if (playBtn) playBtn.disabled = false;
            if (pauseBtn) pauseBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = true;
            if (progressBar) progressBar.value = 0;
            if (timeDisplay) timeDisplay.textContent = '0:00 / 0:00';
        });

        // Handle errors
        audio.addEventListener('error', (e) => {
            console.error('Audio playback error:', e);
            this.addSystemMessage(`❌ Playback error`, 'error');
            if (playBtn) playBtn.disabled = false;
            if (pauseBtn) pauseBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = true;
        });

        // Start playing
        audio.play().then(() => {
            // === TRIGGER NEXT CHAPTER GENERATION ===
            // Send start_chapter message to backend to prefetch the next chapter
            if (this.ws && this.wsConnected && this.currentStoryId) {
                console.log(`📚 Triggering prefetch: Chapter ${chapterNum} playback started`);
                this.ws.send(JSON.stringify({
                    type: 'start_chapter',
                    chapter_number: chapterNum
                }));
                this.addSystemMessage(`📚 Preparing next chapter in background...`, 'info');
            }
        }).catch(err => {
            console.error('Audio play error:', err);
            this.addSystemMessage(`❌ Could not play audio: ${err.message}`, 'error');
            if (playBtn) playBtn.disabled = false;
            if (pauseBtn) pauseBtn.disabled = true;
        });
    }

    pauseChapterTTS(chapterNum) {
        const chapter = chapterNum ?? this.currentChapterTab;
        const audio = this.chapterAudioElements?.[chapter];

        if (audio && !audio.paused) {
            audio.pause();
            const playBtn = document.getElementById(`play-btn-${chapter}`);
            const pauseBtn = document.getElementById(`pause-btn-${chapter}`);
            if (playBtn) playBtn.disabled = false;
            if (pauseBtn) pauseBtn.disabled = true;
            this.addSystemMessage(`⏸️ Paused Chapter ${chapter}`);
        }
    }

    stopChapterTTS(chapterNum) {
        const chapter = chapterNum ?? this.currentChapterTab;
        const audio = this.chapterAudioElements?.[chapter];

        if (audio) {
            audio.pause();
            audio.currentTime = 0;
            const playBtn = document.getElementById(`play-btn-${chapter}`);
            const pauseBtn = document.getElementById(`pause-btn-${chapter}`);
            const stopBtn = document.getElementById(`stop-btn-${chapter}`);
            const progressBar = document.getElementById(`progress-${chapter}`);
            const timeDisplay = document.getElementById(`time-${chapter}`);

            if (playBtn) playBtn.disabled = false;
            if (pauseBtn) pauseBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = true;
            if (progressBar) progressBar.value = 0;
            if (timeDisplay) timeDisplay.textContent = '0:00 / 0:00';

            this.addSystemMessage(`⏹️ Stopped Chapter ${chapter}`);
        }
    }

    seekChapterTTS(chapterNum, value) {
        const chapter = chapterNum ?? this.currentChapterTab;
        const audio = this.chapterAudioElements?.[chapter];

        if (audio && audio.duration) {
            audio.currentTime = (value / 100) * audio.duration;
        }
    }

    formatTime(seconds) {
        if (!seconds || isNaN(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    async fetchCachedAudio(chapterNum) {
        // Try to get audio from the API (which checks Azure Blob Storage)
        const statusEl = document.getElementById(`audio-status-${chapterNum}`);
        if (statusEl) statusEl.textContent = '⏳ Checking for cached audio...';

        try {
            const response = await fetch(
                `${this.apiBaseUrl}/stories/${this.currentStoryId}/chapters/${chapterNum}/generate-audio`,
                { method: 'POST' }
            );

            if (response.ok) {
                const data = await response.json();

                // Cache the result
                this.chapterTTSCache[chapterNum] = {
                    audio: data.fallback_audio_base64,
                    audioUrl: data.audio_url,
                    duration: data.duration_estimate,
                    source: data.source
                };

                // Now play it
                const audioSrc = data.audio_url || (data.fallback_audio_base64
                    ? URL.createObjectURL(this.base64ToBlob(data.fallback_audio_base64, 'audio/mp3'))
                    : null);

                if (audioSrc) {
                    this.setupChapterAudioPlayer(chapterNum, audioSrc);
                    if (statusEl) statusEl.textContent = data.source === 'blob_cache' ? '☁️ Audio cached in Azure' : '✅ Audio ready';
                }
            } else {
                if (statusEl) statusEl.textContent = '❌ No audio available. Generate TTS first.';
                this.addSystemMessage('No audio available. Generate TTS first.', 'error');
            }
        } catch (error) {
            console.error('Error fetching cached audio:', error);
            if (statusEl) statusEl.textContent = '❌ Error fetching audio';
        }
    }

    async regenerateSSML(chapterNum) {
        const chapter = chapterNum ?? this.currentChapterTab;
        const statusEl = document.querySelector(`.ssml-status-row .status-ready, .ssml-status-row .status-pending`);

        this.addSystemMessage(`🔄 Regenerating SSML for Chapter ${chapter}...`);

        try {
            const response = await fetch(
                `${this.apiBaseUrl}/stories/${this.currentStoryId}/chapters/${chapter}/voice-direct?force_regenerate=true`,
                { method: 'POST' }
            );

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `API error: ${response.status}`);
            }

            const data = await response.json();
            this.addSystemMessage(`✅ SSML regenerated for Chapter ${chapter} (${data.ssml_length} chars)`);

            // Refresh the chapter display
            this.selectChapter(chapter);

        } catch (error) {
            console.error('SSML regeneration error:', error);
            this.addSystemMessage(`❌ SSML regeneration failed: ${error.message}`, 'error');
        }
    }

    async regenerateTTS(chapterNum) {
        const chapter = chapterNum ?? this.currentChapterTab;
        const statusEl = document.getElementById(`audio-status-${chapter}`);

        if (statusEl) statusEl.textContent = '🔄 Regenerating audio...';
        this.addSystemMessage(`🔄 Regenerating TTS for Chapter ${chapter}...`);

        try {
            // Call with force_regenerate=true to bypass cache
            const response = await fetch(
                `${this.apiBaseUrl}/stories/${this.currentStoryId}/chapters/${chapter}/generate-audio?force_regenerate=true`,
                { method: 'POST' }
            );

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `API error: ${response.status}`);
            }

            const data = await response.json();

            // Update cache
            this.chapterTTSCache[chapter] = {
                audio: data.fallback_audio_base64,
                audioUrl: data.audio_url,
                duration: data.duration_estimate,
                provider: data.provider,
                source: data.source
            };

            this.addSystemMessage(`✅ Audio regenerated for Chapter ${chapter} (~${data.duration_estimate}s)`);

            // Refresh display
            this.selectChapter(chapter);

        } catch (error) {
            console.error('TTS regeneration error:', error);
            if (statusEl) statusEl.textContent = `❌ Error: ${error.message}`;
            this.addSystemMessage(`❌ TTS regeneration failed: ${error.message}`, 'error');
        }
    }

    downloadChapterTTS(chapterNum) {
        const chapter = chapterNum ?? this.currentChapterTab;
        const ttsData = this.chapterTTSCache[chapter];

        // Try audioUrl first, then base64
        if (ttsData?.audioUrl) {
            // Download from URL
            const a = document.createElement('a');
            a.href = ttsData.audioUrl;
            a.download = `chapter_${chapter}.mp3`;
            a.click();
        } else if (ttsData?.audio) {
            const blob = this.base64ToBlob(ttsData.audio, 'audio/mp3');
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chapter_${chapter}.mp3`;
            a.click();
            URL.revokeObjectURL(url);
        } else {
            this.addSystemMessage('No audio to download', 'error');
        }
    }

    async generateAllTTS() {
        this.addSystemMessage('Generating TTS for all chapters...');

        try {
            const response = await fetch(
                `${this.apiBaseUrl}/stories/${this.currentStoryId}/voice-direct-all`,
                { method: 'POST' }
            );

            if (!response.ok) throw new Error(`API error: ${response.status}`);

            const data = await response.json();

            // Cache all TTS data
            if (data.chapters) {
                data.chapters.forEach(ch => {
                    this.chapterTTSCache[ch.chapter_number] = {
                        audio: ch.audio_base64,
                        ssml: ch.ssml,
                        duration: ch.duration_estimate
                    };
                });
            }

            // Refresh current chapter display
            this.selectChapter(this.currentChapterTab);

            this.addSystemMessage(`TTS generated for ${data.chapters?.length || 0} chapters!`);

        } catch (error) {
            console.error('Generate all TTS error:', error);
            this.addSystemMessage(`Error: ${error.message}`, 'error');
        }
    }

    async refineNorwegian() {
        if (!this.currentStoryId || !this.currentChapterTab) {
            this.addSystemMessage('No chapter selected', 'error');
            return;
        }

        this.addSystemMessage(`🇳🇴 Refining Chapter ${this.currentChapterTab} with Borealis...`);

        try {
            const response = await fetch(
                `${this.apiBaseUrl}/stories/${this.currentStoryId}/chapters/${this.currentChapterTab}/refine-norwegian`,
                { method: 'POST' }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || `API error: ${response.status}`);
            }

            // Store refinement data for comparison
            this.lastRefinementData = {
                chapterNumber: this.currentChapterTab,
                originalContent: data.original_content_full,
                refinedContent: data.refined_content_full,
                originalWordCount: data.original_word_count,
                refinedWordCount: data.refined_word_count,
                wordDiff: data.word_diff,
                wasRefined: data.was_refined
            };

            // Show compare button
            if (this.compareRefinementBtn) {
                this.compareRefinementBtn.classList.remove('hidden');
            }

            if (data.was_refined) {
                this.addSystemMessage(
                    `✅ Refined! ${data.original_word_count} → ${data.refined_word_count} words (${data.word_diff >= 0 ? '+' : ''}${data.word_diff}) — Click 🔍 to compare`
                );
                // Refresh story to show updated content
                await this.fetchStoryDetails(this.currentStoryId);
            } else {
                this.addSystemMessage('ℹ️ No changes made — Click 🔍 to compare and verify');
            }

        } catch (error) {
            console.error('Norwegian refinement error:', error);
            this.addSystemMessage(`Error: ${error.message}`, 'error');
        }
    }

    // ==================== Comparison Modal ====================

    showComparisonModal() {
        // Try to get refinement data from current chapter first, then fall back to lastRefinementData
        let data = null;

        const currentChapter = this.chaptersData?.find(ch => ch.chapter_number === this.currentChapterTab);
        if (currentChapter?.language_refined && currentChapter?.pre_refinement_content) {
            // Use chapter data for comparison
            const originalContent = currentChapter.pre_refinement_content;
            const refinedContent = currentChapter.content;
            const originalWords = originalContent.split(/\s+/).filter(w => w).length;
            const refinedWords = refinedContent.split(/\s+/).filter(w => w).length;

            data = {
                originalContent,
                refinedContent,
                originalWordCount: originalWords,
                refinedWordCount: refinedWords,
                wordDiff: refinedWords - originalWords,
                wasRefined: originalContent !== refinedContent
            };
        } else if (this.lastRefinementData) {
            // Fall back to manual refinement data
            data = this.lastRefinementData;
        }

        if (!data) {
            this.addSystemMessage('No refinement data available. Run refinement first.', 'error');
            return;
        }

        // Update stats
        this.statsOriginalWords.textContent = data.originalWordCount;
        this.statsRefinedWords.textContent = data.refinedWordCount;

        // Word diff
        const diffText = data.wordDiff >= 0 ? `+${data.wordDiff}` : `${data.wordDiff}`;
        this.statsDiff.textContent = diffText;
        this.statsDiff.className = 'stat diff ' + (data.wordDiff > 0 ? 'positive' : data.wordDiff < 0 ? 'negative' : 'neutral');

        // Status
        this.statsStatus.textContent = data.wasRefined ? 'Changed' : 'Unchanged';
        this.statsStatus.className = 'stat status ' + (data.wasRefined ? 'changed' : 'unchanged');

        // Content - format paragraphs
        this.comparisonOriginal.innerHTML = this.formatContentForComparison(data.originalContent);
        this.comparisonRefined.innerHTML = this.formatContentForComparison(data.refinedContent);

        // Store current comparison data for diff view
        this.currentComparisonData = data;

        // Show modal
        this.comparisonModal.classList.remove('hidden');

        // Reset diff view
        this.comparisonOriginal.classList.remove('show-diff');
        this.comparisonRefined.classList.remove('show-diff');
        this.toggleDiffViewBtn.classList.remove('active');
        this.toggleDiffViewBtn.textContent = 'Show Diff Highlights';
    }

    hideComparisonModal() {
        this.comparisonModal.classList.add('hidden');
    }

    formatContentForComparison(content) {
        if (!content) return '<em>No content</em>';

        // Split by double newlines for paragraphs, escape HTML
        const escaped = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        return escaped
            .split(/\n\n+/)
            .filter(p => p.trim())
            .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`)
            .join('');
    }

    toggleDiffView() {
        if (!this.currentComparisonData) return;

        const isShowingDiff = this.comparisonOriginal.classList.contains('show-diff');

        if (isShowingDiff) {
            // Hide diff
            this.comparisonOriginal.classList.remove('show-diff');
            this.comparisonRefined.classList.remove('show-diff');
            this.toggleDiffViewBtn.classList.remove('active');
            this.toggleDiffViewBtn.textContent = 'Show Diff Highlights';

            // Reset to plain content
            this.comparisonOriginal.innerHTML = this.formatContentForComparison(this.currentComparisonData.originalContent);
            this.comparisonRefined.innerHTML = this.formatContentForComparison(this.currentComparisonData.refinedContent);
        } else {
            // Show diff
            this.comparisonOriginal.classList.add('show-diff');
            this.comparisonRefined.classList.add('show-diff');
            this.toggleDiffViewBtn.classList.add('active');
            this.toggleDiffViewBtn.textContent = 'Hide Diff Highlights';

            // Apply word-level diff highlighting
            this.applyDiffHighlighting();
        }
    }

    applyDiffHighlighting() {
        const original = this.currentComparisonData?.originalContent || '';
        const refined = this.currentComparisonData?.refinedContent || '';

        // Split into words while preserving whitespace structure
        const originalWords = original.split(/(\s+)/);
        const refinedWords = refined.split(/(\s+)/);

        // Simple word-by-word comparison
        const originalHighlighted = [];
        const refinedHighlighted = [];

        // Create word sets for quick lookup
        const originalWordSet = new Set(original.toLowerCase().split(/\s+/).filter(w => w));
        const refinedWordSet = new Set(refined.toLowerCase().split(/\s+/).filter(w => w));

        // Highlight words that differ
        for (const word of originalWords) {
            if (word.trim() === '') {
                originalHighlighted.push(word);
            } else {
                const lowerWord = word.toLowerCase().replace(/[.,!?;:"'()]/g, '');
                if (!refinedWordSet.has(lowerWord) && lowerWord) {
                    originalHighlighted.push(`<span class="diff-removed">${this.escapeHtml(word)}</span>`);
                } else {
                    originalHighlighted.push(this.escapeHtml(word));
                }
            }
        }

        for (const word of refinedWords) {
            if (word.trim() === '') {
                refinedHighlighted.push(word);
            } else {
                const lowerWord = word.toLowerCase().replace(/[.,!?;:"'()]/g, '');
                if (!originalWordSet.has(lowerWord) && lowerWord) {
                    refinedHighlighted.push(`<span class="diff-added">${this.escapeHtml(word)}</span>`);
                } else {
                    refinedHighlighted.push(this.escapeHtml(word));
                }
            }
        }

        // Format with paragraphs
        this.comparisonOriginal.innerHTML = originalHighlighted.join('')
            .split(/\n\n+/)
            .filter(p => p.trim())
            .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`)
            .join('');

        this.comparisonRefined.innerHTML = refinedHighlighted.join('')
            .split(/\n\n+/)
            .filter(p => p.trim())
            .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`)
            .join('');
    }

    escapeHtml(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    // ==================== WebSocket ====================

    connectWebSocket(storyId) {
        if (this.ws && this.wsConnected) return;

        // Use dynamic WebSocket URL based on current page location
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/story/${storyId}`;
        console.log('Connecting to WebSocket:', wsUrl);

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.wsConnected = true;
            this.wsStatus.textContent = 'connected';
            this.wsStatus.className = 'value ws-connected';
            console.log('WebSocket connected');

            // If reconnecting to existing story, refresh UI with latest state
            if (this.currentStoryId) {
                console.log('🔄 WebSocket reconnected, refreshing story state...');
                this.fetchStoryDetails(this.currentStoryId);
            }

            this.pingInterval = setInterval(() => {
                if (this.ws?.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, 15000);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('WebSocket message error:', error);
            }
        };

        this.ws.onerror = () => {
            this.wsStatus.textContent = 'error';
            this.wsStatus.className = 'value ws-error';
        };

        this.ws.onclose = (event) => {
            this.wsConnected = false;
            this.wsStatus.textContent = 'disconnected';
            this.wsStatus.className = 'value ws-disconnected';

            if (this.pingInterval) {
                clearInterval(this.pingInterval);
                this.pingInterval = null;
            }

            // Auto-reconnect
            if (this.currentStoryId && event.code !== 1000) {
                setTimeout(() => {
                    if (this.currentStoryId && !this.wsConnected) {
                        this.connectWebSocket(this.currentStoryId);
                    }
                }, 2000);
            }
        };
    }

    handleWebSocketMessage(data) {
        console.log('WebSocket:', data.type, data);

        switch (data.type) {
            case 'connection_established':
                this.addSystemMessage('Connected to story stream');
                break;

            case 'dialogue_ready':
                this.hideTypingIndicator();
                const msg = data.data;
                this.addChatMessage(msg.message, 'narrator', msg.audio);

                // Check for post-chapter discussion
                if (msg.discussion_type === 'post_chapter') {
                    this.discussionIndicator.classList.remove('hidden');
                    this.updatePlaybackPhase('post_chapter');
                }
                break;

            // === STREAMING DIALOGUE EVENTS ===
            // Text chunks arrive immediately for fast feedback

            case 'dialogue_chunk':
                // Append text chunk to streaming message bubble
                this.appendStreamingChunk(data.data.text);
                break;

            case 'dialogue_text_complete':
                // Streaming text is complete - finalize the message
                this.finalizeStreamingMessage(data.data.text);
                break;

            case 'dialogue_audio_ready':
                // Audio arrives after text - add play button to message
                this.addAudioToStreamingMessage(data.data.audio);
                break;

            case 'structure_ready':
                this.addSystemMessage(`Story planned: "${data.data.title}" (${data.data.chapters} chapters)`);
                this.fetchStoryDetails(this.currentStoryId);
                break;

            case 'character_ready':
                this.addSystemMessage(`Character created: ${data.data.name} (${data.data.role})`);
                this.fetchStoryDetails(this.currentStoryId);
                break;

            case 'chapter_ready':
                this.addSystemMessage(`Chapter ${data.data.chapter_number}: "${data.data.title}" ready`);
                this.fetchStoryDetails(this.currentStoryId);
                // === CHAPTER TIMELINE: Mark chapter complete ===
                this.markChapterComplete(data.data.chapter_number);
                break;

            case 'ssml_ready':
                // SSML was auto-generated for this chapter
                this.addSystemMessage(`📝 SSML ready for Chapter ${data.data.chapter_number} (~${data.data.estimated_duration_seconds}s)`);
                // Refresh to update the Generate Audio button state
                this.fetchStoryDetails(this.currentStoryId);
                break;

            case 'audio_tags_generating':
                // Audio tags (ElevenLabs emotion tags) are being generated
                this.addSystemMessage(`🎭 Generating audio tags for Chapter ${data.data.chapter_number}...`);
                break;

            case 'audio_tags_ready':
                // Audio tags are ready - chapter is ready for TTS
                this.addSystemMessage(`✅ Audio tags ready for Chapter ${data.data.chapter_number}`);
                // Refresh to update TTS controls (enables Generate Audio button)
                this.fetchStoryDetails(this.currentStoryId);
                break;

            case 'audio_tags_failed':
                // Audio tag generation failed
                this.addSystemMessage(`⚠️ Audio tag generation failed for Chapter ${data.data.chapter_number}: ${data.data.error || 'Unknown error'}`, 'error');
                break;

            case 'chapter_audio':
                // Audio was auto-generated for resumed story - play it via player controls!
                console.log('🔊 Received chapter_audio event:', data.data);
                this.addSystemMessage(`🔊 Chapter ${data.data.chapter_number} audio ready`);

                // Cache the audio data
                const chNum = data.data.chapter_number;
                this.chapterTTSCache[chNum] = {
                    audio: data.data.audio_base64,
                    audioUrl: data.data.audio_url,  // May be null
                    duration: data.data.duration_estimate,
                    source: data.data.source || 'websocket'
                };

                // If auto_play, use the player controls so progress bar works
                if (data.data.audio_base64 && data.data.auto_play) {
                    const blob = this.base64ToBlob(data.data.audio_base64, 'audio/mp3');
                    const audioSrc = URL.createObjectURL(blob);
                    this.setupChapterAudioPlayer(chNum, audioSrc);
                }

                // Refresh to update UI
                this.fetchStoryDetails(this.currentStoryId);
                break;

            case 'chapter_generating':
                this.addSystemMessage(data.data.message);
                break;

            case 'reading_finished':
                this.updatePlaybackPhase(data.playback_phase || 'post_chapter');
                break;

            case 'input_queued':
                // Only show chapter-specific message if we're actually reading (structure exists)
                // During dialogue phase, this message is confusing since the Storyteller
                // already acknowledges input and ALL context goes to agents anyway
                if (this.story && this.story.structure && data.data.will_appear_in > 1) {
                    this.addSystemMessage(`Your idea will appear in Chapter ${data.data.will_appear_in}!`);
                }
                // If no structure yet, the Storyteller's response is enough - no system message needed
                break;

            case 'error':
                this.addSystemMessage(`Error: ${data.data.message}`, 'error');
                break;

            case 'complete':
                this.addSystemMessage(data.data.message);
                break;

            case 'pong':
                // Keepalive response
                break;

            // ==================== OBSERVATORY EVENTS ====================
            case 'agent_started':
                this.handleAgentStarted(data.data);
                break;

            case 'agent_completed':
                this.handleAgentCompleted(data.data);
                break;

            case 'agent_progress':
                this.handleAgentProgress(data.data);
                break;

            case 'pipeline_stage':
                this.handlePipelineStage(data.data);
                break;

            case 'model_selected':
                this.handleModelSelected(data.data);
                break;

            case 'model_response':
                this.handleModelResponse(data.data);
                break;

            case 'round_table_started':
                this.handleRoundTableStarted(data.data);
                break;

            // reviewer_working removed - now handled by agent_started for unified UX

            case 'reviewer_verdict':
                this.handleReviewerVerdict(data.data);
                break;

            case 'round_table_decision':
                this.handleRoundTableDecision(data.data);
                break;

            case 'max_revisions_exceeded':
                this.handleMaxRevisionsExceeded(data.data);
                break;

            case 'conversation_context':
                // Handled by addEventToLog - just a log event
                break;

            case 'user_input_applied':
                // Handled by addEventToLog - shows user input being incorporated
                break;

            case 'polish_started':
                // === CHAPTER TIMELINE: Update polish phase ===
                if (data.data.chapter) {
                    this.updateTimelinePhase(data.data.chapter, 'polish', 'in_progress');
                }
                break;

            case 'polish_completed':
                // === CHAPTER TIMELINE: Mark polish complete ===
                if (data.data.chapter) {
                    this.updateTimelinePhase(data.data.chapter, 'polish', 'completed');
                }
                break;

            case 'pre_story_inputs_queued':
                // Handled by addEventToLog - shows pre-story messages queued for Ch1
                break;

            // ==================== TTS Pipeline Events ====================
            case 'tts_request_started':
                this.addSystemMessage(`🎤 TTS starting for Ch${data.data.chapter_number} (${data.data.provider})...`);
                break;

            case 'tts_completed': {
                const sizeMB = (data.data.audio_bytes / 1024 / 1024).toFixed(1);
                const storage = data.data.storage === 'blob' ? '☁️ cloud' : '📦 local';
                this.addSystemMessage(`✅ Audio ready: ${sizeMB}MB (~${data.data.duration_estimate}s) [${storage}]`);
                // Refresh to update chapter audio controls
                this.fetchStoryDetails(this.currentStoryId);
                break;
            }

            case 'tts_failed':
                this.addSystemMessage(`❌ TTS failed: ${data.data.error}`, 'error');
                break;
        }

        // Log all events to event log panel
        this.addEventToLog(data);
    }

    // ==================== Observatory Event Handlers ====================

    handleAgentStarted(data) {
        // Track start time for duration calculation
        this.agentStartTimes[data.agent] = Date.now();

        // Check if this is a Round Table reviewer
        const roundTableReviewers = ['Guillermo', 'Bill', 'Clarissa', 'Benjamin', 'Continuity', 'Stephen'];
        if (roundTableReviewers.includes(data.agent)) {
            // Update Round Table panel
            this.updateRoundTableAgent(data.agent, 'working', data.task);

            // Also update Round Table Timeline panel (the reviewer boxes)
            const reviewer = document.querySelector(`[data-reviewer="${data.agent}"]`);
            if (reviewer) {
                reviewer.classList.remove('pending');
                reviewer.classList.add('working');
                reviewer.querySelector('.rt-status').textContent = '●';
            }
        } else {
            // Update production agent panel
            this.updateProductionAgent(data.agent, 'working', data.task);
        }

        // === CHAPTER TIMELINE: Track NarrativeAgent for draft phase ===
        if (data.agent === 'NarrativeAgent' && data.task) {
            const chapterMatch = data.task.match(/Chapter\s*(\d+)/i);
            if (chapterMatch) {
                const chapter = parseInt(chapterMatch[1]);
                this.updateTimelinePhase(chapter, 'draft', 'in_progress');
            }
        }
    }

    handleAgentCompleted(data) {
        // Calculate duration from tracked start time
        const startTime = this.agentStartTimes[data.agent];
        delete this.agentStartTimes[data.agent];

        // Check if this is a Round Table reviewer
        const roundTableReviewers = ['Guillermo', 'Bill', 'Clarissa', 'Benjamin', 'Continuity', 'Stephen'];
        if (roundTableReviewers.includes(data.agent)) {
            // For Round Table reviewers, the reviewer_verdict event will set the final status
            // (approve/concern/block). The agent_completed just shows in Event Log.
            // But we update the panel to show "completed" state briefly
            // The reviewer_verdict handler will then set the final approve/concern/block state
        } else {
            // Return to IDLE state for production agents
            this.updateProductionAgent(data.agent, data.success ? 'idle' : 'error', '');
        }
    }

    handleAgentProgress(data) {
        // Progress updates just keep the agent in working state
        // The simplified UI doesn't show progress percentage
        this.updateProductionAgent(data.agent, 'working', data.step);
    }

    handlePipelineStage(data) {
        const stages = document.getElementById('pipeline-stages');
        if (!stages) return;

        // Map stage names to stage elements
        let stageSelector = data.stage;
        if (data.stage.startsWith('chapter_')) {
            stageSelector = 'chapters';
        }

        const stageEl = stages.querySelector(`[data-stage="${stageSelector}"]`);
        if (!stageEl) return;

        // Remove all status classes
        stageEl.classList.remove('pending', 'active', 'completed', 'error');

        // Add appropriate class based on status
        switch (data.status) {
            case 'in_progress':
                stageEl.classList.add('active');
                stageEl.querySelector('.stage-icon').textContent = '●';
                break;
            case 'completed':
                stageEl.classList.add('completed');
                stageEl.querySelector('.stage-icon').textContent = '✓';
                break;
            case 'error':
                stageEl.classList.add('error');
                stageEl.querySelector('.stage-icon').textContent = '✗';
                break;
            default:
                stageEl.classList.add('pending');
                stageEl.querySelector('.stage-icon').textContent = '○';
        }

        // Update progress if available
        if (data.progress) {
            stageEl.querySelector('.stage-label').textContent =
                `${stageSelector.charAt(0).toUpperCase() + stageSelector.slice(1)} (${data.progress})`;
        }

        // === CHAPTER TIMELINE: Update phase based on pipeline stage ===
        this.updateChapterTimelineFromPipeline(data);
    }

    // ==================== Chapter Timeline ====================

    updateChapterTimelineFromPipeline(data) {
        // Map pipeline stages to timeline phases
        const stage = data.stage;
        const status = data.status;

        // Extract chapter number if this is a chapter-specific stage
        let chapter = 1;  // Default for structure/characters which apply to chapter 1
        const chapterMatch = stage.match(/chapter_(\d+)/);
        if (chapterMatch) {
            chapter = parseInt(chapterMatch[1]);
        }

        // Map stages to timeline phases
        let phase = null;
        if (stage === 'structure') {
            phase = 'structure';
        } else if (stage === 'characters') {
            phase = 'characters';
        } else if (stage.startsWith('chapter_') && stage.includes('draft')) {
            phase = 'draft';
        } else if (stage.startsWith('chapter_') && stage.includes('polish')) {
            phase = 'polish';
        }

        if (phase) {
            this.updateTimelinePhase(chapter, phase, status);
        }
    }

    initChapterTimeline(chapterNum) {
        if (!this.chapterTimelines[chapterNum]) {
            this.chapterTimelines[chapterNum] = {
                phases: {
                    structure: 'waiting',
                    characters: 'waiting',
                    draft: 'waiting',
                    review: 'waiting',
                    polish: 'waiting'
                },
                startTime: new Date(),
                endTime: null
            };
        }
        this.currentTimelineChapter = Math.max(this.currentTimelineChapter, chapterNum);
        this.renderChapterTimeline();
    }

    updateTimelinePhase(chapterNum, phase, status) {
        // Initialize chapter if needed
        if (!this.chapterTimelines[chapterNum]) {
            this.initChapterTimeline(chapterNum);
        }

        const timeline = this.chapterTimelines[chapterNum];

        // Map status to phase state
        let phaseState = 'waiting';
        if (status === 'in_progress') {
            phaseState = 'working';
        } else if (status === 'completed') {
            phaseState = 'done';
        } else if (status === 'error') {
            phaseState = 'error';
        }

        timeline.phases[phase] = phaseState;
        this.renderChapterTimeline();
    }

    markChapterComplete(chapterNum) {
        if (this.chapterTimelines[chapterNum]) {
            this.chapterTimelines[chapterNum].endTime = new Date();
            // Mark all phases as done
            for (const phase of Object.keys(this.chapterTimelines[chapterNum].phases)) {
                if (this.chapterTimelines[chapterNum].phases[phase] !== 'done') {
                    this.chapterTimelines[chapterNum].phases[phase] = 'done';
                }
            }
        }
        this.renderChapterTimeline();
    }

    renderChapterTimeline() {
        const container = document.getElementById('chapter-timeline');
        if (!container) return;

        // If no chapters yet, show empty message
        if (Object.keys(this.chapterTimelines).length === 0) {
            container.innerHTML = '<div class="timeline-empty">Waiting for story generation...</div>';
            return;
        }

        // Build timeline HTML
        let html = '';
        const sortedChapters = Object.keys(this.chapterTimelines).map(Number).sort((a, b) => a - b);

        for (const chapterNum of sortedChapters) {
            const timeline = this.chapterTimelines[chapterNum];
            const phases = timeline.phases;

            // Calculate elapsed time
            let timeDisplay = '';
            if (timeline.startTime) {
                const endTime = timeline.endTime || new Date();
                const elapsed = Math.round((endTime - timeline.startTime) / 1000);
                timeDisplay = elapsed >= 60 ? `${Math.floor(elapsed / 60)}m ${elapsed % 60}s` : `${elapsed}s`;
            }

            html += `
                <div class="timeline-row" data-chapter="${chapterNum}">
                    <span class="timeline-chapter">Ch ${chapterNum}</span>
                    <div class="timeline-phases">
                        <span class="timeline-phase ${phases.structure}" title="Structure">Str</span>
                        <span class="timeline-phase ${phases.characters}" title="Characters">Chr</span>
                        <span class="timeline-phase ${phases.draft}" title="Draft">Drf</span>
                        <span class="timeline-phase ${phases.review}" title="Review">Rev</span>
                        <span class="timeline-phase ${phases.polish}" title="Polish">Pol</span>
                    </div>
                    <span class="timeline-time">${timeDisplay}</span>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    resetChapterTimeline() {
        this.chapterTimelines = {};
        this.currentTimelineChapter = 0;
        this.renderChapterTimeline();
    }

    handleModelSelected(data) {
        const modelName = document.getElementById('current-model');
        const modelMode = document.getElementById('current-mode');

        if (modelName) modelName.textContent = data.model || '--';
        if (modelMode) modelMode.textContent = data.mode || '--';
    }

    handleModelResponse(data) {
        const tokensIn = document.getElementById('tokens-in');
        const tokensOut = document.getElementById('tokens-out');
        const latency = document.getElementById('latency');

        // Accumulate tokens
        if (tokensIn) {
            const current = parseInt(tokensIn.textContent) || 0;
            tokensIn.textContent = (current + (data.tokens_in || 0)).toLocaleString();
        }
        if (tokensOut) {
            const current = parseInt(tokensOut.textContent) || 0;
            tokensOut.textContent = (current + (data.tokens_out || 0)).toLocaleString();
        }
        if (latency && data.latency_ms) {
            latency.textContent = `${(data.latency_ms / 1000).toFixed(1)}s`;
        }
    }

    handleRoundTableStarted(data) {
        console.log('🎯 [DEBUG] round_table_started received:', data);

        const timeline = document.getElementById('roundtable-timeline');
        if (!timeline) {
            console.warn('🎯 [DEBUG] roundtable-timeline element not found!');
            return;
        }

        // === CHAPTER TIMELINE: Update review phase ===
        this.updateTimelinePhase(data.chapter, 'review', 'in_progress');

        // Reviewer descriptions for non-AI experts
        const reviewerDescriptions = {
            'Guillermo': 'Story structure & pacing',
            'Bill': 'Factual accuracy & research',
            'Clarissa': 'Character psychology & authenticity',
            'Benjamin': 'Prose quality & sensory immersion',
            'Continuity': 'Plot thread tracking',
            'Stephen': 'Tension & page-turning hooks'
        };

        timeline.innerHTML = `
            <div class="rt-header">Chapter ${data.chapter} Review</div>
            <div class="rt-reviewers" id="rt-reviewers">
                ${data.reviewers.map(r => `
                    <div class="rt-reviewer pending" data-reviewer="${r}" title="${reviewerDescriptions[r] || 'Reviewer'}">
                        <span class="rt-status">⏳</span>
                        <span class="rt-name">${r}</span>
                        <span class="rt-domain">${reviewerDescriptions[r] || ''}</span>
                    </div>
                `).join('')}
            </div>
            <div class="rt-decision" id="rt-decision">Reviewing...</div>
        `;

        // === UPDATE ACTIVE AGENTS PANEL: Set all to pending (will change to working as each starts) ===
        const roundTableAgents = ['Guillermo', 'Bill', 'Clarissa', 'Benjamin', 'Continuity', 'Stephen'];
        roundTableAgents.forEach(agent => {
            this.updateRoundTableAgent(agent, 'pending', `Ch${data.chapter}`);
        });
    }

    handleReviewerWorking(data) {
        // Change from pending (⏳) to working (● blinking)
        const reviewer = document.querySelector(`[data-reviewer="${data.reviewer}"]`);
        if (reviewer) {
            reviewer.classList.remove('pending');
            reviewer.classList.add('working');
            reviewer.querySelector('.rt-status').textContent = '●';
        }

        // Update Agents panel too
        this.updateRoundTableAgent(data.reviewer, 'working', `Ch${data.chapter}`);
    }

    handleReviewerVerdict(data) {
        const reviewer = document.querySelector(`[data-reviewer="${data.reviewer}"]`);
        if (!reviewer) return;

        // Remove pending and working states
        reviewer.classList.remove('pending', 'working');

        switch (data.verdict) {
            case 'approve':
                reviewer.classList.add('approved');
                reviewer.querySelector('.rt-status').textContent = '✅';
                break;
            case 'concern':
                reviewer.classList.add('concern');
                reviewer.querySelector('.rt-status').textContent = '⚠️';
                break;
            case 'block':
                reviewer.classList.add('blocked');
                reviewer.querySelector('.rt-status').textContent = '🚫';
                break;
        }

        // Add duration badge if available
        if (data.duration_ms) {
            const existingDuration = reviewer.querySelector('.rt-duration');
            if (existingDuration) existingDuration.remove();

            const durationSec = (data.duration_ms / 1000).toFixed(1);
            const durationBadge = document.createElement('span');
            durationBadge.className = 'rt-duration';
            durationBadge.textContent = `${durationSec}s`;
            // Insert after the name
            const nameEl = reviewer.querySelector('.rt-name');
            if (nameEl) {
                nameEl.insertAdjacentElement('afterend', durationBadge);
            }
        }

        // === UPDATE ACTIVE AGENTS PANEL ===
        this.updateRoundTableAgent(data.reviewer, data.verdict, data.notes ? data.notes.substring(0, 30) : data.verdict);

        // Show notes prominently for any verdict that has notes
        // (reviewers can approve with minor concerns that are still worth displaying)
        if (data.notes) {
            // Add has-notes class for vertical layout (matches concern/blocked styling)
            reviewer.classList.add('has-notes');

            // Remove any existing notes element
            const existingNotes = reviewer.querySelector('.rt-notes');
            if (existingNotes) existingNotes.remove();

            const notesEl = document.createElement('div');
            notesEl.className = 'rt-notes';  // CSS styling comes from parent class (.approved, .concern, .blocked)
            notesEl.textContent = data.notes;
            reviewer.appendChild(notesEl);
        }

        // Add explanation for Benjamin blocks (non-AI expert friendly)
        if (data.reviewer === 'Benjamin' && data.verdict === 'block') {
            const existingExplanation = reviewer.querySelector('.rt-explanation');
            if (!existingExplanation) {
                const explanation = document.createElement('div');
                explanation.className = 'rt-explanation';
                explanation.innerHTML = `
                    <strong>What this means:</strong> Benjamin checks if every paragraph includes
                    sensory details (sight, sound, touch, smell). When prose lacks these immersive
                    elements, young readers may lose engagement.
                `;
                reviewer.appendChild(explanation);
            }
        }

        // Keep hover title for full text
        if (data.notes) {
            reviewer.title = data.notes;
        }
    }

    updateRoundTableAgent(agentName, status, task) {
        // Find the agent in the Round Table agents list
        const agentsList = document.getElementById('roundtable-agents-list');
        if (!agentsList) {
            console.warn(`🎯 [DEBUG] roundtable-agents-list not found when updating ${agentName}`);
            return;
        }

        const agentItem = agentsList.querySelector(`[data-agent="${agentName}"]`);
        if (!agentItem) {
            console.warn(`🎯 [DEBUG] Agent element not found: data-agent="${agentName}"`);
            return;
        }

        // Agent emoji map
        const emojiMap = {
            'Guillermo': '🎬', 'Bill': '🔬', 'Clarissa': '📚',
            'Benjamin': '✏️', 'Continuity': '🔗', 'Stephen': '⚡'
        };
        const emoji = emojiMap[agentName] || '👤';

        // Remove all status classes
        agentItem.classList.remove('idle', 'working', 'approve', 'concern', 'block', 'completed');

        // Update based on status
        let statusIcon = '○';
        if (status === 'working') {
            agentItem.classList.add('working');
            statusIcon = '◉';
        } else if (status === 'approve') {
            agentItem.classList.add('approve');
            statusIcon = '✓';
        } else if (status === 'concern') {
            agentItem.classList.add('concern');
            statusIcon = '⚠';
        } else if (status === 'block') {
            agentItem.classList.add('block');
            statusIcon = '✗';
        } else {
            agentItem.classList.add('idle');
        }

        // Update text content
        agentItem.textContent = `${statusIcon} ${emoji} ${agentName}`;
    }

    resetRoundTableAgents() {
        const roundTableAgents = ['Guillermo', 'Bill', 'Clarissa', 'Benjamin', 'Continuity', 'Stephen'];
        roundTableAgents.forEach(agent => {
            this.updateRoundTableAgent(agent, 'idle', '');
        });
    }

    updateProductionAgent(agentName, status, task) {
        const agentsList = document.getElementById('agents-list');
        if (!agentsList) return;

        const agentItem = agentsList.querySelector(`[data-agent="${agentName}"]`);
        if (!agentItem) return;

        const emojiMap = {
            'StructureAgent': '📐', 'CharacterAgent': '🎭', 'NarrativeAgent': '✍️',
            'VoiceDirectorAgent': '🎙️', 'CompanionAgent': '🤝', 'LineEditorAgent': '💅'
        };
        const nameMap = {
            'StructureAgent': 'Structure', 'CharacterAgent': 'Character', 'NarrativeAgent': 'Narrative',
            'VoiceDirectorAgent': 'Voice', 'CompanionAgent': 'Companion', 'LineEditorAgent': 'Polish'
        };
        const emoji = emojiMap[agentName] || '🔧';
        const name = nameMap[agentName] || agentName;

        agentItem.classList.remove('idle', 'working', 'completed', 'error');

        let statusIcon = '○';
        if (status === 'working') {
            agentItem.classList.add('working');
            statusIcon = '◉';
        } else if (status === 'completed') {
            agentItem.classList.add('completed');
            statusIcon = '✓';
        } else if (status === 'error') {
            agentItem.classList.add('error');
            statusIcon = '✗';
        } else {
            agentItem.classList.add('idle');
        }

        agentItem.textContent = `${statusIcon} ${emoji} ${name}`;
    }

    handleRoundTableDecision(data) {
        const decision = document.getElementById('rt-decision');
        if (!decision) return;

        if (data.decision === 'approved') {
            decision.innerHTML = `<span class="decision-approved">✅ Approved</span>`;
            decision.classList.add('approved');
            // === CHAPTER TIMELINE: Mark review as complete ===
            if (data.chapter) {
                this.updateTimelinePhase(data.chapter, 'review', 'completed');
            }
            // Reset Round Table agents to idle after approval (with slight delay for visual feedback)
            setTimeout(() => this.resetRoundTableAgents(), 2000);
        } else {
            decision.innerHTML = `<span class="decision-revision">🔄 Revision Round ${data.revision_round}</span>`;
            decision.classList.add('revision');
            // Keep agents showing their verdicts during revision
        }
    }

    handleMaxRevisionsExceeded(data) {
        const decision = document.getElementById('rt-decision');
        if (decision) {
            decision.innerHTML = `
                <span class="decision-warning">
                    ⚠️ Max Revisions (${data.revisions}) - Proceeding with Warning
                </span>
                <div class="max-revisions-details">
                    <span class="blockers-label">Blockers:</span>
                    <span class="blockers-list">${data.blockers?.join(', ') || 'none'}</span>
                </div>
            `;
            decision.classList.add('warning');
        }

        // Show system message in chat
        this.addSystemMessage(
            `⚠️ Chapter ${data.chapter} exceeded max revision attempts. ${data.message}`,
            'warning'
        );
    }

    createAgentItem(agentName, task, status) {
        const agentsList = document.getElementById('agents-list');
        if (!agentsList) return;

        const statusIcon = status === 'working' ? '●' : status === 'completed' ? '✓' : '○';

        const item = document.createElement('div');
        item.className = `agent-item ${status}`;
        item.dataset.agent = agentName;
        item.innerHTML = `
            <span class="agent-status">${statusIcon}</span>
            <span class="agent-name">${agentName}</span>
            <span class="agent-task">${task}</span>
        `;
        agentsList.appendChild(item);
    }

    addEventToLog(event) {
        const eventLog = document.getElementById('event-log');
        const eventCount = document.getElementById('event-count');
        if (!eventLog) return;

        // Remove empty message
        const empty = eventLog.querySelector('.event-empty');
        if (empty) empty.remove();

        // Create event entry
        const time = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const entry = document.createElement('div');
        entry.className = `event-entry event-${event.type}`;

        // Create header (clickable for expand/collapse)
        const header = document.createElement('div');
        header.className = 'event-header';
        header.innerHTML = `
            <span class="event-time">${time}</span>
            <span class="event-type">${event.type}</span>
            <span class="event-summary">${this.summarizeEvent(event)}</span>
        `;
        entry.appendChild(header);

        // Add click to HEADER only (not details) - allows text selection in details
        header.addEventListener('click', () => {
            if (entry.classList.contains('expanded')) {
                entry.classList.remove('expanded');
                entry.querySelector('.event-details')?.remove();
            } else {
                entry.classList.add('expanded');
                const details = document.createElement('pre');
                details.className = 'event-details';
                details.textContent = JSON.stringify(event.data, null, 2);
                // Stop clicks on details from closing the entry
                details.addEventListener('click', (e) => e.stopPropagation());
                entry.appendChild(details);
            }
        });

        // Prepend to log (newest first)
        eventLog.prepend(entry);

        // Update count
        if (eventCount) {
            const count = eventLog.querySelectorAll('.event-entry').length;
            eventCount.textContent = count;
        }

        // NOTE: No longer limiting log size - user wants to keep all events for debugging
        // Use the "Clear Log" button in Debug controls to manually clear if needed
    }

    clearEventLog() {
        const eventLog = document.getElementById('event-log');
        const eventCount = document.getElementById('event-count');
        if (eventLog) {
            eventLog.innerHTML = '<div class="event-empty">Log cleared. Waiting for events...</div>';
        }
        if (eventCount) {
            eventCount.textContent = '0';
        }
    }

    summarizeEvent(event) {
        const d = event.data || {};
        switch (event.type) {
            case 'agent_started': {
                const instanceInfo = d.instance ? ` (${d.instance}/${d.total_instances})` : '';
                // Note: model is NOT shown in agent_started (we don't know it yet)
                return d.agent + instanceInfo + ': ' + (d.task || '').substring(0, 30);
            }
            case 'agent_completed': {
                const instanceInfo = d.instance ? ` (${d.instance}/${d.total_instances})` : '';
                const modelSuffix = d.model ? ` [${d.model}]` : '';
                // Show error reason when failed
                const errorInfo = !d.success && d.error ? ` - ${d.error.substring(0, 50)}` : '';
                return d.agent + instanceInfo + (d.success ? ' ✓' : ' ✗') + errorInfo + modelSuffix;
            }
            case 'pipeline_stage': return d.stage + ' → ' + d.status;
            case 'model_selected': return d.model + ' (' + d.mode + ')';
            case 'model_response': return d.tokens_in + '→' + d.tokens_out + ' tokens';
            case 'round_table_started': return 'Ch' + d.chapter + ' review';
            // reviewer_working removed - now using agent_started instead
            case 'reviewer_verdict': {
                const modelInfo = d.model ? ` [${d.model.replace('azure/', '')}]` : '';
                const durationInfo = d.duration_ms ? ` (${(d.duration_ms / 1000).toFixed(1)}s)` : '';
                return `${d.reviewer}: ${d.verdict}${durationInfo}${modelInfo}`;
            }
            case 'round_table_decision': return d.decision;
            case 'structure_ready': return d.title || 'Structure ready';
            case 'character_ready': return d.name || 'Character ready';
            case 'chapter_ready': return 'Ch' + d.chapter_number;
            case 'conversation_context':
                const speaker = d.speaker === 'user' ? '👤' : '🎭';
                const name = d.name || d.speaker;
                const msg = (d.message || '').substring(0, 40);
                return `${speaker} ${name}: ${msg}${d.message?.length > 40 ? '...' : ''}`;
            case 'max_revisions_exceeded': {
                const parts = [];
                if (d.blockers?.length) parts.push(`Blockers: ${d.blockers.join(', ')}`);
                if (d.concerns?.length) parts.push(`Concerns: ${d.concerns.join(', ')}`);
                const status = parts.length ? parts.join('; ') : 'proceeding';
                return `⚠️ Ch${d.chapter} after ${d.revisions} rounds (${status})`;
            }
            case 'user_input_applied':
                return `📝 ${d.inputs_count} input(s) → Ch${d.chapter}`;
            case 'polish_started': {
                const agentInfo = d.agent ? ` [${d.agent}]` : '';
                return `✨ Polishing Ch${d.chapter} (${d.suggestions_count} suggestions)${agentInfo}`;
            }
            case 'polish_completed': {
                const modelInfo = d.model ? ` [${d.model.replace('azure/', '')}]` : '';
                const diffSign = d.word_diff >= 0 ? '+' : '';
                return `✨ Polish done: ${d.word_count_before || '?'}→${d.word_count_after || '?'} (${diffSign}${d.word_diff || '?'})${modelInfo}`;
            }
            case 'pre_story_inputs_queued':
                return `📋 ${d.count} pre-story input(s) queued for Ch${d.target_chapter}`;
            case 'audio_tags_generating':
                return `🎭 Ch${d.chapter_number} audio tags generating`;
            case 'audio_tags_ready':
                return `✅ Ch${d.chapter_number} audio tags ready`;
            case 'audio_tags_failed':
                return `⚠️ Ch${d.chapter_number} audio tags failed`;
            default: return '';
        }
    }

    // ==================== Chat ====================

    async handleChatInput() {
        const text = this.chatInput.value.trim();
        if (!text) return;

        this.addChatMessage(text, 'user');
        this.chatInput.value = '';

        // If we're in exploring/greeting/ask_preference mode (no story yet), use conversation/continue
        // This covers: exploring, greeting, or choosing between multiple stories
        const needsConversationContinue = !this.currentStoryId && this.pendingIntent && (
            ['exploring', 'greeting'].includes(this.pendingIntent) ||
            this.pendingSuggestedAction === 'ask_preference'
        );

        if (needsConversationContinue) {
            this.conversationTurn++;
            await this.continueExploringConversation(text);
            return;
        }

        // Otherwise, send via WebSocket for active story
        if (this.ws && this.wsConnected) {
            this.ws.send(JSON.stringify({
                type: 'user_message',
                message: text
            }));
            this.showTypingIndicator();
        } else {
            this.addSystemMessage('Not connected. Create a story first.', 'error');
        }
    }

    async continueExploringConversation(text) {
        /**
         * Continue exploring conversation via /conversation/continue endpoint.
         * Used when child is deciding what story they want.
         */
        try {
            this.showTypingIndicator();

            const response = await fetch(`${this.apiBaseUrl}/conversation/continue?conversation_turn=${this.conversationTurn}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    child_id: this.selectedChildId,
                    language: this.parentLanguage
                })
            });

            this.hideTypingIndicator();

            if (!response.ok) throw new Error(`API error: ${response.status}`);

            const convData = await response.json();

            // Track pre-story user messages for Chapter 1 context
            this.preStoryMessages.push(text);

            // Show CompanionAgent response
            this.addChatMessage(convData.dialogue, 'narrator', convData.audio);

            // Update intent state
            this.pendingIntent = convData.intent;
            this.pendingActiveStoryId = convData.active_story_id;
            this.pendingSuggestedAction = convData.suggested_action;

            console.log(`🎯 Conversation turn ${this.conversationTurn}: ${convData.intent}, action: ${convData.suggested_action}`);

            // Route based on new intent
            if (convData.intent === 'continue' && convData.suggested_action === 'resume_story' && convData.active_story_id) {
                // Child decided to continue - resume existing story
                await this.resumeExistingStory(convData.active_story_id);
            } else if (convData.intent === 'new_story' || convData.suggested_action === 'init_story') {
                // Child decided on new story - proceed to story generation
                await this.initializeNewStory(text);
            }
            // Otherwise, stay in exploring mode - user can continue chatting

        } catch (error) {
            this.hideTypingIndicator();
            console.error('Error continuing conversation:', error);
            this.addSystemMessage(`Error: ${error.message}`, 'error');
        }
    }

    addChatMessage(message, sender, audio = null) {
        // Guard against undefined/null messages
        if (!message || message === 'undefined') {
            console.warn('Skipping undefined chat message');
            return;
        }

        const welcome = this.chatHistory.querySelector('.chat-welcome');
        if (welcome) welcome.remove();

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message message-${sender}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        bubble.innerHTML = `
            <div class="message-sender">${sender === 'user' ? 'You' : '🎭 Storyteller'}</div>
            <div class="message-text">${message}</div>
            ${audio ? `<button class="audio-play-btn" onclick="debugPanel.playAudioBase64('${audio.substring(0, 100)}...')">🔊 Play</button>` : ''}
            <div class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
        `;

        // Store full audio for playback
        if (audio) {
            const playBtn = bubble.querySelector('.audio-play-btn');
            playBtn.onclick = () => this.playAudioBase64(audio);
        }

        messageDiv.appendChild(bubble);
        this.chatHistory.appendChild(messageDiv);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;

        // Queue audio for playback (uses audio queue to prevent overlapping)
        if (audio && sender === 'narrator') {
            this.playAudioBase64(audio);
        }
    }

    // ==================== Streaming Dialogue Helpers ====================
    // Handle streaming dialogue - text arrives chunk by chunk, audio follows

    appendStreamingChunk(text) {
        // Get or create the streaming message bubble
        if (!this.streamingMessageBubble) {
            // Hide typing indicator since text is arriving
            this.hideTypingIndicator();

            const welcome = this.chatHistory.querySelector('.chat-welcome');
            if (welcome) welcome.remove();

            // Create new message container
            const messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message message-narrator';
            messageDiv.id = 'streaming-message';

            const bubble = document.createElement('div');
            bubble.className = 'message-bubble streaming';

            bubble.innerHTML = `
                <div class="message-sender">🎭 Storyteller</div>
                <div class="message-text"></div>
                <div class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
            `;

            messageDiv.appendChild(bubble);
            this.chatHistory.appendChild(messageDiv);
            this.streamingMessageBubble = bubble;
        }

        // Append chunk to the message text
        const textDiv = this.streamingMessageBubble.querySelector('.message-text');
        if (textDiv) {
            textDiv.textContent += text;
        }

        // Scroll to keep message visible
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    finalizeStreamingMessage(fullText) {
        if (this.streamingMessageBubble) {
            // Update with full text (in case any chunks were missed)
            const textDiv = this.streamingMessageBubble.querySelector('.message-text');
            if (textDiv && fullText) {
                textDiv.textContent = fullText;
            }

            // Remove streaming class (stops any animation)
            this.streamingMessageBubble.classList.remove('streaming');

            // Keep reference for audio attachment
            this.lastStreamingBubble = this.streamingMessageBubble;
            this.streamingMessageBubble = null;
        }
    }

    addAudioToStreamingMessage(audioBase64) {
        if (this.lastStreamingBubble && audioBase64) {
            // Add play button if not already present
            if (!this.lastStreamingBubble.querySelector('.audio-play-btn')) {
                const timeDiv = this.lastStreamingBubble.querySelector('.message-time');
                const playBtn = document.createElement('button');
                playBtn.className = 'audio-play-btn';
                playBtn.textContent = '🔊 Play';
                playBtn.onclick = () => this.playAudioBase64(audioBase64);

                // Insert before time
                if (timeDiv) {
                    this.lastStreamingBubble.insertBefore(playBtn, timeDiv);
                } else {
                    this.lastStreamingBubble.appendChild(playBtn);
                }
            }

            // Auto-play the audio
            this.playAudioBase64(audioBase64);
        }

        // Clear reference
        this.lastStreamingBubble = null;
    }

    addSystemMessage(text, type = 'info') {
        const welcome = this.chatHistory.querySelector('.chat-welcome');
        if (welcome) welcome.remove();

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message message-system ${type === 'error' ? 'message-error' : ''}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble system-bubble';
        bubble.textContent = text;

        messageDiv.appendChild(bubble);
        this.chatHistory.appendChild(messageDiv);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    showTypingIndicator() {
        this.hideTypingIndicator();
        const typing = document.createElement('div');
        typing.id = 'typing-indicator';
        typing.className = 'chat-message message-narrator';
        typing.innerHTML = '<div class="message-bubble typing-indicator"><span></span><span></span><span></span></div>';
        this.chatHistory.appendChild(typing);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }

    hideTypingIndicator() {
        document.getElementById('typing-indicator')?.remove();
    }

    // ==================== Audio ====================

    playAudioBase64(audioBase64) {
        this.audioQueue.push(audioBase64);
        if (!this.isPlayingAudio) this.playNextAudio();
    }

    playNextAudio() {
        if (this.audioQueue.length === 0) {
            this.isPlayingAudio = false;
            this.stopSpeechBtn.disabled = true;
            return;
        }

        const audioBase64 = this.audioQueue.shift();
        this.isPlayingAudio = true;

        try {
            const blob = this.base64ToBlob(audioBase64, 'audio/mp3');
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);

            this.stopSpeechBtn.disabled = false;
            this.currentAudio = audio;

            audio.onended = () => {
                URL.revokeObjectURL(url);
                this.currentAudio = null;
                this.playNextAudio();
            };

            audio.onerror = () => {
                this.currentAudio = null;
                this.playNextAudio();
            };

            audio.play().catch(() => this.playNextAudio());

        } catch (error) {
            console.error('Audio error:', error);
            this.playNextAudio();
        }
    }

    tryAutoplayAudio(audioBase64) {
        try {
            const blob = this.base64ToBlob(audioBase64, 'audio/mp3');
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.play().catch(() => console.log('Autoplay blocked'));
            audio.onended = () => URL.revokeObjectURL(url);
        } catch (e) {
            console.log('Autoplay error:', e.message);
        }
    }

    stopAudio() {
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio = null;
        }
        this.audioQueue = [];
        this.isPlayingAudio = false;
        this.stopSpeechBtn.disabled = true;
    }

    // ==================== Playback Phase ====================

    updatePlaybackPhase(phase) {
        this.playbackPhase = phase;
        const badge = this.playbackPhaseBadge;
        if (badge) {
            badge.textContent = phase.toUpperCase().replace('_', ' ');
            badge.className = `playback-phase-badge phase-${phase}`;
        }

        // Update discussion indicator
        if (phase === 'post_chapter') {
            this.discussionIndicator?.classList.remove('hidden');
        } else {
            this.discussionIndicator?.classList.add('hidden');
        }
    }

    // ==================== Utilities ====================

    base64ToBlob(base64, mimeType) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return new Blob([bytes], { type: mimeType });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    updateMicStatus(text, state) {
        this.micStatus.textContent = text;
    }

    clearAll() {
        this.storyInput.value = '';
        this.chatInput.value = '';
        this.currentStoryId = null;
        this.storyData = null;
        this.chaptersData = [];
        this.chapterTTSCache = {};

        // Reset conversation state
        this.conversationTurn = 0;
        this.pendingIntent = null;
        this.pendingActiveStoryId = null;
        this.pendingSuggestedAction = null;

        // Reset chapter timeline
        this.resetChapterTimeline();

        // Reset profile selections
        this.selectedChildId = null;
        this.selectedChild = null;
        document.querySelectorAll('.child-profile-card').forEach(c => c.classList.remove('selected'));

        this.storyDetailsPanel.classList.add('hidden');
        this.chaptersPanel.classList.add('hidden');
        this.chatInputContainer.classList.add('hidden');
        this.discussionIndicator?.classList.add('hidden');
        this.activeProfileSummary?.classList.add('hidden');
        this.storyCreationPanel?.classList.add('hidden');

        this.chatHistory.innerHTML = '<div class="chat-welcome">👋 Select a profile and create a story to start...</div>';
        this.rawJson.textContent = 'No story loaded';

        this.stopAudio();

        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.wsConnected = false;
        }

        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }

        // Clear persisted session
        this.clearSession();
    }

    checkBrowserSupport() {
        const srSupport = document.getElementById('sr-support');
        const ssSupport = document.getElementById('ss-support');

        const hasSR = 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
        const hasSS = 'speechSynthesis' in window;

        if (srSupport) {
            srSupport.textContent = hasSR ? '✅ Yes' : '❌ No';
            srSupport.style.color = hasSR ? 'green' : 'red';
        }

        if (ssSupport) {
            ssSupport.textContent = hasSS ? '✅ Yes' : '❌ No';
            ssSupport.style.color = hasSS ? 'green' : 'red';
        }
    }

    async checkApiStatus() {
        const statusEl = document.getElementById('api-status');
        statusEl.textContent = 'checking...';

        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (response.ok) {
                statusEl.textContent = '✅ Connected';
                statusEl.style.color = 'green';
            } else {
                throw new Error('Not healthy');
            }
        } catch (error) {
            statusEl.textContent = '❌ Not connected';
            statusEl.style.color = 'red';
        }
    }
}

// Initialize
let debugPanel;
document.addEventListener('DOMContentLoaded', async () => {
    debugPanel = new DebugPanel();
    window.debugPanel = debugPanel;  // Make accessible globally for onclick handlers

    // Try to restore previous session (parent/child/story selection)
    await debugPanel.tryRestoreSession();
});
