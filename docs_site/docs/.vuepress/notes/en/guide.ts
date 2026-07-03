import type { ThemeNote } from 'vuepress-theme-plume'
import { defineNoteConfig } from 'vuepress-theme-plume'

export const enGuide: ThemeNote = defineNoteConfig({
  dir: 'guide',
  link: '/guide/',
  sidebar: [
    {
      text: 'Basic Info',
      collapsed: false,
      icon: 'carbon:idea',
      prefix: 'basicinfo',
      items: ['intro', 'framework', 'install'],
    },
    {
      text: 'Reweighter',
      collapsed: false,
      icon: 'solar:scale-outline',
      prefix: 'reweighter',
      items: ['quickstart', 'tutorial', 'advantage_reweight'],
    },
    {
      text: 'Selector',
      collapsed: false,
      icon: 'solar:filter-outline',
      prefix: 'selector',
      items: ['quickstart', 'tutorial', 'difficulty_filtering'],
    },
    {
      text: 'Mixer',
      collapsed: false,
      icon: 'solar:shuffle-outline',
      prefix: 'mixer',
      items: ['quickstart', 'tutorial', 'dump_ucb'],
    },
  ],
})
