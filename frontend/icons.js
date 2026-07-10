/* Hand-drawn monoline SVG icons. 24x24 viewBox, ~1.8-2px stroke, round caps/joins,
   currentColor stroke so CSS can tint per style. No icon library, no emoji. */

const ICONS = {
  // formal -> document
  formal: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M6.5 3h7l4 4v13a1 1 0 0 1-1 1h-10a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/>
    <path d="M13.5 3v4h4"/>
    <line x1="8.5" y1="11.5" x2="15.5" y2="11.5"/>
    <line x1="8.5" y1="14.7" x2="15.5" y2="14.7"/>
    <line x1="8.5" y1="17.9" x2="12.5" y2="17.9"/>
  </svg>`,

  // sarcastic -> smirk face
  sarcastic: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="8.5"/>
    <circle cx="9" cy="10.2" r="0.9" fill="currentColor" stroke="none"/>
    <circle cx="15" cy="10.2" r="0.9" fill="currentColor" stroke="none"/>
    <path d="M8.3 14.8c1.1 1.1 3 1.6 4.6 1.1 1.5-.5 2.5-1.4 2.8-2.1"/>
  </svg>`,

  // humorous_tech -> code bracket "</>"
  humorous_tech: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="9 7 4 12 9 17"/>
    <polyline points="15 7 20 12 15 17"/>
  </svg>`,

  // humorous_non_tech -> laugh face
  humorous_non_tech: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="8.5"/>
    <path d="M8.4 9.6c.5-.75 1.3-.75 1.8 0"/>
    <path d="M13.8 9.6c.5-.75 1.3-.75 1.8 0"/>
    <path d="M7.5 13.6c.9 2.1 2.9 3.4 4.5 3.4s3.6-1.3 4.5-3.4"/>
  </svg>`,

  // play glyph for VideoCard thumbnails
  play: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="9"/>
    <path d="M10.2 8.3l6 3.7-6 3.7v-7.4z" fill="currentColor" stroke="none"/>
  </svg>`,

  // upload glyph for the dropzone
  upload: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 16V4"/>
    <polyline points="7 9 12 4 17 9"/>
    <path d="M4.5 16v3a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1v-3"/>
  </svg>`,

  // link glyph for the URL input
  link: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M9.5 14.5l5-5"/>
    <path d="M11 8.5l1.2-1.2a3 3 0 0 1 4.24 4.24L15 12.9"/>
    <path d="M13 15.5l-1.2 1.2a3 3 0 0 1-4.24-4.24L8.9 11.1"/>
  </svg>`,
};
