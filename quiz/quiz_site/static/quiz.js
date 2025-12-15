// quiz_site/static/quiz.js

// ========================================
// Dane quizów - łatwe pytania dla 10-latka
// ========================================

const quizData = {
    car: {
        title: "🚗 Quiz Samochodowy",
        questions: [
            {
                id: 1,
                type: "radio",
                text: "Jakie światło na sygnalizacji oznacza STOP?",
                options: ["Zielone", "Żółte", "Czerwone", "Niebieskie"],
                correct: "Czerwone"
            },
            {
                id: 2,
                type: "checkbox",
                text: "Zaznacz wszystkie pojazdy, które mają koła:",
                options: ["Samochód", "Łódka", "Rower", "Samolot", "Motocykl"],
                correct: ["Samochód", "Rower", "Motocykl"]
            },
            {
                id: 3,
                type: "dropdown",
                text: "Ile kół ma typowy samochód osobowy?",
                options: ["Wybierz odpowiedź", "2", "3", "4", "6"],
                correct: "4"
            },
            {
                id: 4,
                type: "text",
                text: "Jak nazywa się osoba, która kieruje samochodem?",
                placeholder: "Wpisz odpowiedź...",
                correct: ["kierowca", "szofer", "kierowcą"],
                hint: "Podpowiedź: zaczyna się na K"
            },
            {
                id: 5,
                type: "radio",
                text: "Po której stronie drogi jeżdżą samochody w Polsce?",
                options: ["Po lewej", "Po prawej", "Środkiem", "Chodnikiem"],
                correct: "Po prawej"
            },
            {
                id: 6,
                type: "checkbox",
                text: "Co powinno być w każdym samochodzie? (wybierz wszystkie poprawne)",
                options: ["Gaśnica", "Trójkąt ostrzegawczy", "Telewizor", "Apteczka", "Basen"],
                correct: ["Gaśnica", "Trójkąt ostrzegawczy", "Apteczka"]
            },
            {
                id: 7,
                type: "dropdown",
                text: "Czym napędzane są samochody elektryczne?",
                options: ["Wybierz odpowiedź", "Benzyną", "Węglem", "Elektrycznością", "Wodą"],
                correct: "Elektrycznością"
            },
            {
                id: 8,
                type: "radio",
                text: "Co oznacza znak STOP?",
                options: [
                    "Jedź szybciej",
                    "Zatrzymaj się",
                    "Skręć w lewo",
                    "Jedź prosto"
                ],
                correct: "Zatrzymaj się"
            },
            {
                id: 9,
                type: "text",
                text: "Jak nazywa się miejsce, gdzie tankujemy samochód?",
                placeholder: "Wpisz odpowiedź...",
                correct: ["stacja benzynowa", "stacja paliw", "stacja"],
                hint: "Podpowiedź: stacja ..."
            },
            {
                id: 10,
                type: "radio",
                text: "Gdzie powinni przechodzić piesi przez ulicę?",
                options: [
                    "Gdziekolwiek",
                    "Na przejściu dla pieszych (zebrze)",
                    "Pod samochodami",
                    "Na czerwonym świetle"
                ],
                correct: "Na przejściu dla pieszych (zebrze)"
            }
        ]
    },
    
    life: {
        title: "🌍 Quiz o Życiu",
        questions: [
            {
                id: 1,
                type: "radio",
                text: "Ile dni ma tydzień?",
                options: ["5", "6", "7", "10"],
                correct: "7"
            },
            {
                id: 2,
                type: "checkbox",
                text: "Które z tych są owocami? (zaznacz wszystkie)",
                options: ["Jabłko", "Marchewka", "Banan", "Ziemniak", "Pomarańcza"],
                correct: ["Jabłko", "Banan", "Pomarańcza"]
            },
            {
                id: 3,
                type: "dropdown",
                text: "Ile miesięcy ma rok?",
                options: ["Wybierz odpowiedź", "10", "11", "12", "13"],
                correct: "12"
            },
            {
                id: 4,
                type: "text",
                text: "Jak nazywa się stolica Polski?",
                placeholder: "Wpisz nazwę miasta...",
                correct: ["warszawa", "Warszawa"],
                hint: "Podpowiedź: zaczyna się na W"
            },
            {
                id: 5,
                type: "radio",
                text: "Jakie zwierzę mówi 'hau hau'?",
                options: ["Kot", "Pies", "Krowa", "Kura"],
                correct: "Pies"
            },
            {
                id: 6,
                type: "checkbox",
                text: "Zaznacz wszystkie kolory tęczy:",
                options: ["Czerwony", "Czarny", "Żółty", "Szary", "Niebieski"],
                correct: ["Czerwony", "Żółty", "Niebieski"]
            },
            {
                id: 7,
                type: "dropdown",
                text: "Która pora roku jest najcieplejsza?",
                options: ["Wybierz odpowiedź", "Wiosna", "Lato", "Jesień", "Zima"],
                correct: "Lato"
            },
            {
                id: 8,
                type: "radio",
                text: "Co daje nam Słońce?",
                options: ["Deszcz", "Światło i ciepło", "Śnieg", "Wiatr"],
                correct: "Światło i ciepło"
            },
            {
                id: 9,
                type: "text",
                text: "Ile nóg ma pająk?",
                placeholder: "Wpisz liczbę...",
                correct: ["8", "osiem"],
                hint: "Podpowiedź: więcej niż 6"
            },
            {
                id: 10,
                type: "radio",
                text: "Z czego zrobiony jest lód?",
                options: ["Z ognia", "Z kamieni", "Z wody", "Z powietrza"],
                correct: "Z wody"
            }
        ]
    },
    
    shopping: {
        title: "🛒 Quiz Zakupowy",
        questions: [
            {
                id: 1,
                type: "radio",
                text: "Gdzie kupujemy chleb?",
                options: ["W aptece", "W piekarni", "Na poczcie", "W kinie"],
                correct: "W piekarni"
            },
            {
                id: 2,
                type: "checkbox",
                text: "Co można kupić w sklepie spożywczym? (zaznacz wszystkie)",
                options: ["Mleko", "Telewizor", "Jajka", "Samochód", "Ser"],
                correct: ["Mleko", "Jajka", "Ser"]
            },
            {
                id: 3,
                type: "dropdown",
                text: "Ile groszy jest w jednej złotówce?",
                options: ["Wybierz odpowiedź", "10", "50", "100", "1000"],
                correct: "100"
            },
            {
                id: 4,
                type: "text",
                text: "Jak nazywa się osoba, która sprzedaje w sklepie?",
                placeholder: "Wpisz odpowiedź...",
                correct: ["sprzedawca", "ekspedient", "sprzedawczyni"],
                hint: "Podpowiedź: zaczyna się na S"
            },
            {
                id: 5,
                type: "radio",
                text: "Co robimy z koszykiem w sklepie?",
                options: [
                    "Rzucamy nim",
                    "Wkładamy do niego produkty",
                    "Siedzimy w nim",
                    "Jemy go"
                ],
                correct: "Wkładamy do niego produkty"
            },
            {
                id: 6,
                type: "checkbox",
                text: "Które z tych rzeczy są nabiałem?",
                options: ["Jogurt", "Chleb", "Masło", "Mąka", "Śmietana"],
                correct: ["Jogurt", "Masło", "Śmietana"]
            },
            {
                id: 7,
                type: "dropdown",
                text: "Gdzie płacimy za zakupy?",
                options: ["Wybierz odpowiedź", "Przy wejściu", "W magazynie", "Przy kasie", "Na parkingu"],
                correct: "Przy kasie"
            },
            {
                id: 8,
                type: "radio",
                text: "Co dostajemy po zapłaceniu za zakupy?",
                options: ["Prezent", "Paragon", "Medal", "Certyfikat"],
                correct: "Paragon"
            },
            {
                id: 9,
                type: "text",
                text: "Jak nazywa się duży sklep z wieloma działami?",
                placeholder: "Wpisz odpowiedź...",
                correct: ["supermarket", "hipermarket", "market", "centrum handlowe"],
                hint: "Podpowiedź: super..."
            },
            {
                id: 10,
                type: "radio",
                text: "Co powinniśmy zrobić przed jedzeniem owoców ze sklepu?",
                options: [
                    "Schować je",
                    "Umyć je",
                    "Pomalować je",
                    "Zamrozić je"
                ],
                correct: "Umyć je"
            }
        ]
    }
};

// ========================================
// Zmienne globalne
// ========================================

let currentQuiz = null;
let currentQuestionIndex = 0;
let userAnswers = {};
let timerInterval = null;
let elapsedSeconds = 0;
let cookiesAccepted = false;

// ========================================
// Inicjalizacja
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    initCookies();
    
    // Sprawdź czy jesteśmy na stronie quizu
    const path = window.location.pathname;
    if (path.startsWith('/quiz/')) {
        const quizType = path.split('/')[2];
        if (quizData[quizType]) {
            initQuiz(quizType);
        } else {
            window.location.href = '/';
        }
    }
    
    // Aktualizuj aktywny link w menu
    updateActiveNavLink();
    updateStats();
});

// ========================================
// Obsługa cookies
// ========================================

function initCookies() {
    const cookieBanner = document.getElementById('cookieBanner');
    const acceptBtn = document.getElementById('acceptCookies');
    const declineBtn = document.getElementById('declineCookies');
    
    // Sprawdź czy użytkownik już wybrał
    const cookieChoice = localStorage.getItem('cookiesAccepted');
    
    if (cookieChoice === null) {
        cookieBanner?.classList.remove('hidden');
    } else {
        cookiesAccepted = cookieChoice === 'true';
    }
    
    acceptBtn?.addEventListener('click', function() {
        localStorage.setItem('cookiesAccepted', 'true');
        cookiesAccepted = true;
        cookieBanner?.classList.add('hidden');
    });
    
    declineBtn?.addEventListener('click', function() {
        localStorage.setItem('cookiesAccepted', 'false');
        cookiesAccepted = false;
        cookieBanner?.classList.add('hidden');
    });
}

// ========================================
// Inicjalizacja quizu
// ========================================

function initQuiz(quizType) {
    currentQuiz = quizData[quizType];
    currentQuestionIndex = 0;
    userAnswers = {};
    
    // Ustaw tytuł
    document.getElementById('quizTitle').textContent = currentQuiz.title;
    document.getElementById('quizType').textContent = quizType.charAt(0).toUpperCase() + quizType.slice(1);
    document.title = currentQuiz.title;
    
    // Renderuj pytania
    renderQuestions();
    
    // Ustaw przyciski nawigacji
    setupNavigation();
    
    // Uruchom timer
    startTimer();
    
    // Obsługa wyszukiwarki pytań
    setupQuestionSearch();
    
    // Pokaż pierwsze pytanie
    showQuestion(0);
}

function renderQuestions() {
    const container = document.getElementById('questionsContainer');
    container.innerHTML = '';
    
    currentQuiz.questions.forEach((question, index) => {
        const questionEl = document.createElement('div');
        questionEl.className = 'question-block';
        questionEl.id = `question-${index}`;
        questionEl.dataset.questionId = question.id;
        
        let typeLabel = '';
        switch(question.type) {
            case 'radio': typeLabel = 'Wybór pojedynczy'; break;
            case 'checkbox': typeLabel = 'Wielokrotny wybór'; break;
            case 'dropdown': typeLabel = 'Lista rozwijana'; break;
            case 'text': typeLabel = 'Pytanie otwarte'; break;
        }
        
        questionEl.innerHTML = `
            <span class="question-number">Pytanie ${index + 1} z ${currentQuiz.questions.length}</span>
            <p class="question-text">
                ${question.text}
                <span class="question-type-badge">${typeLabel}</span>
            </p>
            <div class="answer-container">
                ${renderAnswerInput(question, index)}
            </div>
        `;
        
        container.appendChild(questionEl);
    });
}

function renderAnswerInput(question, index) {
    switch(question.type) {
        case 'radio':
            return renderRadioOptions(question, index);
        case 'checkbox':
            return renderCheckboxOptions(question, index);
        case 'dropdown':
            return renderDropdown(question, index);
        case 'text':
            return renderTextInput(question, index);
        default:
            return '';
    }
}

function renderRadioOptions(question, index) {
    return `
        <div class="options-list">
            ${question.options.map((option, optIndex) => `
                <div class="option-item">
                    <input type="radio" 
                           id="q${index}_opt${optIndex}" 
                           name="question_${index}" 
                           value="${option}"
                           onchange="saveAnswer(${index}, '${option.replace(/'/g, "\\'")}')">
                    <label class="option-label" for="q${index}_opt${optIndex}">
                        <span class="option-indicator"></span>
                        ${option}
                    </label>
                </div>
            `).join('')}
        </div>
    `;
}

function renderCheckboxOptions(question, index) {
    return `
        <div class="options-list">
            ${question.options.map((option, optIndex) => `
                <div class="option-item">
                    <input type="checkbox" 
                           id="q${index}_opt${optIndex}" 
                           name="question_${index}" 
                           value="${option}"
                           onchange="saveCheckboxAnswer(${index})">
                    <label class="option-label" for="q${index}_opt${optIndex}">
                        <span class="option-indicator"></span>
                        ${option}
                    </label>
                </div>
            `).join('')}
        </div>
        <p class="input-hint">Możesz wybrać więcej niż jedną odpowiedź</p>
    `;
}

function renderDropdown(question, index) {
    return `
        <div class="dropdown-container">
            <select class="dropdown-select" 
                    id="dropdown_${index}" 
                    onchange="saveAnswer(${index}, this.value)">
                ${question.options.map(option => `
                    <option value="${option}">${option}</option>
                `).join('')}
            </select>
        </div>
    `;
}

function renderTextInput(question, index) {
    return `
        <div class="text-input-container">
            <input type="text" 
                   class="text-input" 
                   id="text_${index}"
                   placeholder="${question.placeholder || 'Wpisz odpowiedź...'}"
                   oninput="saveAnswer(${index}, this.value)"
                   autocomplete="off">
            ${question.hint ? `<p class="input-hint">${question.hint}</p>` : ''}
        </div>
    `;
}

// ========================================
// Zapisywanie odpowiedzi
// ========================================

function saveAnswer(questionIndex, value) {
    userAnswers[questionIndex] = value;
    updateProgress();
}

function saveCheckboxAnswer(questionIndex) {
    const checkboxes = document.querySelectorAll(`input[name="question_${questionIndex}"]:checked`);
    const values = Array.from(checkboxes).map(cb => cb.value);
    userAnswers[questionIndex] = values;
    updateProgress();
}

// ========================================
// Nawigacja między pytaniami
// ========================================

function setupNavigation() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitBtn');
    const form = document.getElementById('quizForm');
    
    prevBtn?.addEventListener('click', () => navigateQuestion(-1));
    nextBtn?.addEventListener('click', () => navigateQuestion(1));
    
    form?.addEventListener('submit', function(e) {
        e.preventDefault();
        submitQuiz();
    });
    
    document.getElementById('retryBtn')?.addEventListener('click', function() {
        window.location.reload();
    });
}

function navigateQuestion(direction) {
    const newIndex = currentQuestionIndex + direction;
    if (newIndex >= 0 && newIndex < currentQuiz.questions.length) {
        showQuestion(newIndex);
    }
}

function showQuestion(index) {
    currentQuestionIndex = index;
    
    // Ukryj wszystkie pytania
    document.querySelectorAll('.question-block').forEach(q => {
        q.classList.remove('active');
    });
    
    // Pokaż aktualne pytanie
    document.getElementById(`question-${index}`)?.classList.add('active');
    
    // Aktualizuj przyciski
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitBtn');
    
    prevBtn.disabled = index === 0;
    
    if (index === currentQuiz.questions.length - 1) {
        nextBtn.classList.add('hidden');
        submitBtn.classList.remove('hidden');
    } else {
        nextBtn.classList.remove('hidden');
        submitBtn.classList.add('hidden');
    }
    
    // Aktualizuj licznik
    document.getElementById('questionCounter').textContent = 
        `${index + 1}/${currentQuiz.questions.length}`;
    
    updateProgress();
}

function updateProgress() {
    const total = currentQuiz.questions.length;
    const answered = Object.keys(userAnswers).length;
    const percent = (answered / total) * 100;
    
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = 
        `Odpowiedziano: ${answered} z ${total}`;
}

// ========================================
// Timer
// ========================================

function startTimer() {
    elapsedSeconds = 0;
    updateTimerDisplay();
    
    timerInterval = setInterval(() => {
        elapsedSeconds++;
        updateTimerDisplay();
    }, 1000);
}

function updateTimerDisplay() {
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = elapsedSeconds % 60;
    document.getElementById('timerDisplay').textContent = 
        `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// ========================================
// Wyszukiwarka pytań
// ========================================

function setupQuestionSearch() {
    const searchInput = document.getElementById('questionSearch');
    const goBtn = document.getElementById('goToQuestion');
    
    goBtn?.addEventListener('click', function() {
        const num = parseInt(searchInput.value);
        if (num >= 1 && num <= currentQuiz.questions.length) {
            showQuestion(num - 1);
            searchInput.value = '';
        }
    });
    
    searchInput?.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            goBtn.click();
        }
    });
}

// ========================================
// Wysyłanie quizu i wyświetlanie wyników
// ========================================

function submitQuiz() {
    stopTimer();
    
    // Oblicz wynik
    const results = calculateResults();
    
    // Pokaż panel wyników
    displayResults(results);
    
    // Wyślij dane do serwera
    sendResults(results);
    
    // Zapisz statystyki
    saveStats(results);
}

function calculateResults() {
    let correct = 0;
    const details = [];
    
    currentQuiz.questions.forEach((question, index) => {
        const userAnswer = userAnswers[index];
        let isCorrect = false;
        
        switch(question.type) {
            case 'radio':
            case 'dropdown':
                isCorrect = userAnswer === question.correct;
                break;
            case 'checkbox':
                if (Array.isArray(userAnswer) && Array.isArray(question.correct)) {
                    isCorrect = arraysEqual(userAnswer.sort(), question.correct.sort());
                }
                break;
            case 'text':
                if (userAnswer && Array.isArray(question.correct)) {
                    isCorrect = question.correct.some(
                        c => c.toLowerCase() === userAnswer.toLowerCase().trim()
                    );
                }
                break;
        }
        
        if (isCorrect) correct++;
        
        details.push({
            question: question.text,
            userAnswer: formatAnswer(userAnswer),
            correctAnswer: formatAnswer(question.correct),
            isCorrect
        });
    });
    
    return {
        correct,
        total: currentQuiz.questions.length,
        percent: Math.round((correct / currentQuiz.questions.length) * 100),
        time: elapsedSeconds,
        details
    };
}

function formatAnswer(answer) {
    if (Array.isArray(answer)) {
        return answer.join(', ');
    }
    return answer || 'Brak odpowiedzi';
}

function arraysEqual(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

function displayResults(results) {
    // Ukryj formularz
    document.getElementById('quizForm').classList.add('hidden');
    
    // Pokaż panel wyników
    const resultsPanel = document.getElementById('resultsPanel');
    resultsPanel.classList.remove('hidden');
    
    // Ustaw wynik
    document.getElementById('scorePercent').textContent = `${results.percent}%`;
    document.getElementById('scoreText').textContent = 
        `Poprawnych odpowiedzi: ${results.correct}/${results.total} (czas: ${formatTime(results.time)})`;
    
    // Pokaż szczegóły
    const detailsContainer = document.getElementById('resultsDetails');
    detailsContainer.innerHTML = results.details.map((detail, index) => `
        <div class="result-item ${detail.isCorrect ? 'correct' : 'incorrect'}">
            <span class="result-icon">${detail.isCorrect ? '✅' : '❌'}</span>
            <div class="result-content">
                <p class="result-question">${index + 1}. ${detail.question}</p>
                <p class="result-answer">Twoja odpowiedź: ${detail.userAnswer}</p>
                ${!detail.isCorrect ? `<p class="result-correct-answer">Poprawna: ${detail.correctAnswer}</p>` : ''}
            </div>
        </div>
    `).join('');
    
    // Przewiń do góry
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
}

async function sendResults(results) {
    const quizType = window.location.pathname.split('/')[2];
    
    const data = {
        quizType,
        results: {
            correct: results.correct,
            total: results.total,
            percent: results.percent,
            time: results.time
        },
        answers: userAnswers,
        cookiesAccepted
    };
    
    try {
        await fetch('/api/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
    } catch (error) {
        console.error('Error submitting results:', error);
    }
}

// ========================================
// Statystyki
// ========================================

function saveStats(results) {
    if (!cookiesAccepted) return;
    
    const stats = JSON.parse(localStorage.getItem('quizStats') || '{"taken": 0, "bestScore": 0}');
    stats.taken++;
    if (results.percent > stats.bestScore) {
        stats.bestScore = results.percent;
    }
    localStorage.setItem('quizStats', JSON.stringify(stats));
}

function updateStats() {
    const stats = JSON.parse(localStorage.getItem('quizStats') || '{"taken": 0, "bestScore": 0}');
    
    const takenEl = document.querySelector('#quizzesTaken span');
    const bestEl = document.querySelector('#bestScore span');
    
    if (takenEl) takenEl.textContent = stats.taken;
    if (bestEl) bestEl.textContent = stats.taken > 0 ? `${stats.bestScore}%` : '-';
}

// ========================================
// Aktywny link w menu
// ========================================

function updateActiveNavLink() {
    const path = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === path) {
            link.classList.add('active');
        }
    });
}

// Eksport funkcji dla HTML
window.saveAnswer = saveAnswer;
window.saveCheckboxAnswer = saveCheckboxAnswer;
window.initCookies = initCookies;
window.updateStats = updateStats;