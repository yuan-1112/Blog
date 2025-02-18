import { createApp } from 'vue'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'
import App from './App.vue'

// 处理 ResizeObserver 循环错误
const debounceRAF = (callback) => {
  let rafId = null;
  let lastEntries = [];
  
  return (...args) => {
    if (rafId) {
      cancelAnimationFrame(rafId);
    }
    
    // 保存最新的 entries
    if (args[0] && args[0].length) {
      lastEntries = args[0];
    }
    
    rafId = requestAnimationFrame(() => {
      rafId = null;
      if (lastEntries.length) {
        callback(lastEntries);
        lastEntries = [];
      }
    });
  };
};

// 重写 ResizeObserver 以优化性能
if (typeof window !== 'undefined') {
  const OrignalResizeObserver = window.ResizeObserver;
  
  window.ResizeObserver = class ResizeObserver extends OrignalResizeObserver {
    constructor(callback) {
      const wrappedCallback = (entries) => {
        try {
          callback(entries);
        } catch (e) {
          // 忽略特定的 ResizeObserver 错误
          if (!e.message.includes('ResizeObserver loop')) {
            console.error(e);
          }
        }
      };
      
      super(debounceRAF(wrappedCallback));
    }
  };
  
  // 防止错误显示在控制台
  const originalError = console.error;
  console.error = (...args) => {
    if (args[0] && typeof args[0] === 'string' && args[0].includes('ResizeObserver loop')) {
      return;
    }
    originalError.apply(console, args);
  };
  
  // 阻止错误事件冒泡 - 同时处理 error 和 unhandledrejection 事件
  const errorHandler = (e) => {
    if (e.message?.includes?.('ResizeObserver loop') || 
        e.reason?.message?.includes?.('ResizeObserver loop')) {
      e.stopPropagation();
      e.preventDefault();
      return false;
    }
  };
  
  window.addEventListener('error', errorHandler, true);
  window.addEventListener('unhandledrejection', errorHandler, true);
}

const app = createApp(App)
app.use(Antd)
app.mount('#app')
