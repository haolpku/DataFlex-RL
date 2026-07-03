import { defineNotesConfig } from 'vuepress-theme-plume'
import { enGuide } from './guide.js'
export const enNotes = defineNotesConfig({ dir: '/en/notes/', link: '/en/', notes: [enGuide] })
