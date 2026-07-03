import { defineNotesConfig } from 'vuepress-theme-plume'
import { zhGuide } from './guide.js'
export const zhNotes = defineNotesConfig({ dir: '/zh/notes/', link: '/zh/', notes: [zhGuide] })
