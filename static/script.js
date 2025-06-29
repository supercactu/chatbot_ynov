 // Gestion du formulaire de chatbot
document.getElementById('chatbot-form').onsubmit = async function(e) {
    e.preventDefault();
    
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.querySelector('.loading');
    const submitBtn = document.querySelector('.submit-btn');
    
    // Afficher le loading
    loadingDiv.classList.add('show');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Traitement...';
    resultDiv.classList.remove('show');
    
    try {
        const formData = new FormData(e.target);
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        const result = await response.text();
        
        // Simuler un délai pour l'effet visuel
        setTimeout(() => {
            resultDiv.innerHTML = result;
            resultDiv.classList.add('show');
            loadingDiv.classList.remove('show');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Envoyer';
        }, 500);
        
    } catch (error) {
        setTimeout(() => {
            resultDiv.innerHTML = '<p style="color: #ff6b6b;">❌ Erreur lors du traitement de votre demande.</p>';
            resultDiv.classList.add('show');
            loadingDiv.classList.remove('show');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Envoyer';
        }, 500);
    }
};

// Animation au scroll pour les éléments
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.animationPlayState = 'running';
        }
    });
}, observerOptions);

// Appliquer l'observateur à tous les éléments form-group
document.querySelectorAll('.form-group').forEach(el => {
    observer.observe(el);
});

// Fonction utilitaire pour afficher des messages de notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        background: ${type === 'error' ? '#ff6b6b' : type === 'success' ? '#51cf66' : '#339af0'};
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    `;
    
    document.body.appendChild(notification);
    
    // Animation d'entrée
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Suppression automatique après 3 secondes
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Validation du formulaire avant soumission
function validateForm() {
    const textarea = document.querySelector('textarea[name="user_input"]');
    const select = document.querySelector('select[name="task"]');
    
    if (!textarea.value.trim()) {
        showNotification('Veuillez entrer du texte avant de soumettre.', 'error');
        textarea.focus();
        return false;
    }
    
    if (textarea.value.trim().length < 3) {
        showNotification('Le texte doit contenir au moins 3 caractères.', 'error');
        textarea.focus();
        return false;
    }
    
    return true;
}

// Améliorer la soumission du formulaire avec validation
document.getElementById('chatbot-form').addEventListener('submit', function(e) {
    if (!validateForm()) {
        e.preventDefault();
        return false;
    }
});

// Ajouter des raccourcis clavier
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter pour soumettre le formulaire
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (validateForm()) {
            document.getElementById('chatbot-form').dispatchEvent(new Event('submit'));
        }
    }
    
    // Échap pour effacer le textarea
    if (e.key === 'Escape') {
        const textarea = document.querySelector('textarea[name="user_input"]');
        if (document.activeElement === textarea) {
            textarea.value = '';
            showNotification('Texte effacé.', 'info');
        }
    }
});

// Auto-resize du textarea
const textarea = document.querySelector('textarea[name="user_input"]');
textarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.max(120, this.scrollHeight) + 'px';
});

// Compteur de caractères (optionnel)
function addCharacterCounter() {
    const textarea = document.querySelector('textarea[name="user_input"]');
    const counter = document.createElement('div');
    counter.className = 'char-counter';
    counter.style.cssText = `
        position: absolute;
        bottom: 8px;
        right: 15px;
        font-size: 12px;
        color: #666;
        background: rgba(255,255,255,0.8);
        padding: 2px 6px;
        border-radius: 4px;
    `;
    
    textarea.parentElement.style.position = 'relative';
    textarea.parentElement.appendChild(counter);
    
    function updateCounter() {
        const count = textarea.value.length;
        counter.textContent = `${count} caractères`;
        counter.style.color = count > 1000 ? '#ff6b6b' : '#666';
    }
    
    textarea.addEventListener('input', updateCounter);
    updateCounter();
}

// Initialiser le compteur de caractères au chargement de la page
document.addEventListener('DOMContentLoaded', addCharacterCounter);