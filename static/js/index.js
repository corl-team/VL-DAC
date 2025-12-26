// Emoji celebration animation - optimized physics & performance
const EMOJIS = Object.freeze(['🎉', '🎊', '📚', '🔬', '📖', '✨', '🏆', '🚀', '⭐', '🎓', '💡', '🔥', '💪', '👏', '🙌']);
const CONFIG = Object.freeze({
  count: 30,
  spawnInterval: 50,
  fallDuration: { min: 2.5, max: 4 },
  fontSize: { min: 1.6, max: 2.8 },
  startY: { min: -8, max: -3 },
  horizontalRange: { min: 8, max: 92 },
  drift: { min: -60, max: 60 },
  rotation: { min: 180, max: 720 },
  wobbleAmp: { min: 8, max: 20 },
  wobbleFreq: { min: 3, max: 6 }
});

function rand(min, max) {
  return min + Math.random() * (max - min);
}

function createEmojiCelebration() {
  const container = document.createElement('div');
  container.className = 'emoji-container';
  document.body.appendChild(container);

  const { count, spawnInterval, fallDuration, fontSize, startY, horizontalRange, drift, rotation, wobbleAmp, wobbleFreq } = CONFIG;

  // Pre-generate all emoji data
  const emojiData = Array.from({ length: count }, () => ({
    emoji: EMOJIS[(Math.random() * EMOJIS.length) | 0],
    left: rand(horizontalRange.min, horizontalRange.max),
    startYVal: rand(startY.min, startY.max),
    size: rand(fontSize.min, fontSize.max),
    duration: rand(fallDuration.min, fallDuration.max),
    driftVal: rand(drift.min, drift.max),
    rotateVal: rand(rotation.min, rotation.max) * (Math.random() > 0.5 ? 1 : -1),
    wobbleAmpVal: rand(wobbleAmp.min, wobbleAmp.max),
    wobbleFreqVal: rand(wobbleFreq.min, wobbleFreq.max)
  }));

  let spawned = 0;
  let active = 0;
  let startTime = null;

  function spawnEmoji(data) {
    const el = document.createElement('div');
    el.className = 'falling-emoji';
    el.textContent = data.emoji;

    // Set all CSS variables at once
    el.style.cssText = `
      left: ${data.left}%;
      font-size: ${data.size}rem;
      --start-y: ${data.startYVal}vh;
      --drift: ${data.driftVal}px;
      --rotate: ${data.rotateVal}deg;
      --wobble-amp: ${data.wobbleAmpVal}px;
      --wobble-freq: ${data.wobbleFreqVal};
      animation-duration: ${data.duration}s;
    `;

    active++;
    el.addEventListener('animationend', () => {
      el.remove();
      if (--active === 0 && spawned === count) container.remove();
    }, { once: true });

    container.appendChild(el);
  }

  function tick(timestamp) {
    if (!startTime) startTime = timestamp;
    const elapsed = timestamp - startTime;

    // Spawn emojis based on elapsed time
    while (spawned < count && elapsed >= spawned * spawnInterval) {
      spawnEmoji(emojiData[spawned]);
      spawned++;
    }

    if (spawned < count) {
      requestAnimationFrame(tick);
    }
  }

  requestAnimationFrame(tick);
}

// Copy BibTeX functionality
document.addEventListener('DOMContentLoaded', () => {
  const button = document.getElementById('copy-bibtex-btn');
  const bibtexContent = document.getElementById('bibtex-content');
  const copyText = button?.querySelector('.copy-text');
  const icon = button?.querySelector('.icon i');

  if (!button || !bibtexContent || !copyText || !icon) return;

  let timeoutId;

  const resetButton = () => {
    button.classList.remove('copied');
    copyText.textContent = 'Copy';
    icon.className = 'fas fa-copy';
  };

  // Fallback for browsers without Clipboard API (older Safari/iOS)
  const fallbackCopy = (text) => {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.cssText = 'position:fixed;left:-9999px;top:-9999px';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    let success = false;
    try {
      success = document.execCommand('copy');
    } catch {
      success = false;
    }

    document.body.removeChild(textarea);
    return success;
  };

  const copyToClipboard = async (text) => {
    // Try modern Clipboard API first
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch {
        // Fall through to fallback
      }
    }
    // Fallback for iOS Safari and older browsers
    return fallbackCopy(text);
  };

  button.addEventListener('click', async () => {
    const success = await copyToClipboard(bibtexContent.textContent.trim());
    clearTimeout(timeoutId);

    if (success) {
      button.classList.add('copied');
      copyText.textContent = 'Copied!';
      icon.className = 'fas fa-check';
      createEmojiCelebration();
    } else {
      copyText.textContent = 'Failed';
    }

    timeoutId = setTimeout(resetButton, 2000);
  });
});
