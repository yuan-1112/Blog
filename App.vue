<script>
import FormDesigner from './views/FormDesigner.vue'
import axios from 'axios'
import { message } from 'ant-design-vue';

export default {
  name: 'App',
  components: {
    FormDesigner
  },
  methods: {
    //保存表单时，前端会发送一个 POST 请求到后端的 /api/forms 路由
    //POST /api/forms 请求会调用 FormController 的 CreateForm 方法来处理表单创建
    async saveForm() {
      try {
        const response = await api.post('/forms', {
          title: this.formTitle,
          config: JSON.stringify(this.formConfig),
        });
        console.log('Form saved:', response.data);
      } catch (error) {
        console.error('Error saving form:', error);
      }
    },

    async submitForm(formData) {
      try {
        const response = await api.post(`/forms/${this.currentFormId}/submit`, {
          form_id: this.currentFormId,
          data: JSON.stringify(formData),
        });
        console.log('Form submitted:', response.data);
      } catch (error) {
        console.error('Error submitting form:', error);
      }
    },
  }
}

// 修改 API 配置
//前端使用 axios 库来发送 HTTP 请求到后端 API。
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// 添加请求拦截器用于调试
api.interceptors.request.use(config => {
  console.log('Making request:', config);
  return config;
});

// 添加响应拦截器用于调试
//后端处理完请求后，会返回 JSON 格式的响应给前端。
//前端通过 axios 的响应拦截器来处理这些响应
api.interceptors.response.use(
  response => {
    console.log('Response:', response);
    return response;
  },
  error => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// 设置网页标题
document.title = '表单设计器'
</script>

<template>
  <form-designer />
</template>

<style>
html {
  scroll-behavior: smooth;
  height: 100%;
}

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial,
    'Noto Sans', sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol',
    'Noto Color Emoji';
  height: 100%;
  overflow-y: overlay;
}

#app {
  height: 100%;
  margin: 0;
  padding: 0;
}

* {
  box-sizing: border-box;
}

/* 优化滚动条样式 */
::-webkit-scrollbar {
  width: 8px;
  background-color: transparent;
}

::-webkit-scrollbar-thumb {
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
  transition: background-color 0.3s;
}

::-webkit-scrollbar-thumb:hover {
  background-color: rgba(0, 0, 0, 0.3);
}

::-webkit-scrollbar-track {
  background-color: transparent;
}

.smooth-scroll {
  scroll-behavior: smooth;
  -webkit-overflow-scrolling: touch;
  will-change: scroll-position;
  transform: translateZ(0);
}
</style>
