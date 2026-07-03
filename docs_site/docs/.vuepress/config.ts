import { viteBundler } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import { plumeTheme } from 'vuepress-theme-plume'
import { enNavbar, zhNavbar } from './navbars/index.js'
import { enNotes, zhNotes } from './notes/index.js'

export default defineUserConfig({
  base: '/DataFlex-RL/',
  locales: {
    '/en/': { lang: 'en-US', title: 'DataFlex-RL', description: 'Data scheduling for RL fine-tuning (verl plugin)' },
    '/zh/': { lang: 'zh-CN', title: 'DataFlex-RL', description: 'RL 微调的数据调度(verl 插件)' },
  },
  bundler: viteBundler(),
  theme: plumeTheme({
    hostname: 'https://haolpku.github.io',
    docsDir: 'docs_site/docs',
    social: [{ icon: 'github', link: 'https://github.com/haolpku/DataFlex-RL' }],
    locales: {
      '/en/': { navbar: enNavbar, notes: enNotes },
      '/zh/': { navbar: zhNavbar, notes: zhNotes },
    },
  }),
})
