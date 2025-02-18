<template>
  <div class="form-designer">
    <a-layout>
      <a-layout-header class="header">
        <div class="header-content">
          <div class="logo">表单设计器</div>
          <div class="actions">
            <a-space>
              <a-button type="primary" @click="generateFormWithAI">
                <template #icon><thunderbolt-outlined /></template>
                AI生成表单
              </a-button>
              <a-button type="primary" @click="saveForm">保存</a-button>
              <a-button @click="exportForm">导出</a-button>
              <a-button type="primary" @click="previewForm">预览</a-button>
              <a-upload accept=".json" :show-upload-list="false" :before-upload="importForm">
                <a-button>导入</a-button>
              </a-upload>
              <a-popconfirm
                title="确定要清空所有表单项吗？"
                @confirm="clearAllForms"
                ok-text="确定"
                cancel-text="取消"
              >
                <a-button danger>清空</a-button>
              </a-popconfirm>
            </a-space>
          </div>
        </div>
      </a-layout-header>
      
      <a-layout-content class="content">
        <a-layout class="inner-layout">
          <a-layout-sider width="250" class="sider">
            <form-item-library @add-component="handleAddComponent" />
          </a-layout-sider>
          
          <a-layout-content class="main-content">
            <design-area
              ref="designArea"
              @item-selected="onItemSelected"
            />
          </a-layout-content>
          
          <a-layout-sider width="300" class="sider">
            <property-panel :selected-item="selectedItem" />
          </a-layout-sider>
        </a-layout>
      </a-layout-content>
    </a-layout>

    <!-- AI表单生成对话框 -->
    <a-modal
      v-model:open="aiModalVisible"
      title="AI表单生成"
      @ok="handleAIFormGenerate"
      @cancel="handleAIFormCancel"
      :confirmLoading="aiGenerating"
    >
      <a-form layout="vertical">
        <a-form-item label="请描述你想要生成的表单：">
          <a-textarea
            v-model:value="aiPrompt"
            :rows="4"
            placeholder="例如：创建一个用户注册表单，包含用户名、邮箱和密码字段"
          />
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 预览对话框 -->
    <a-modal
      v-model:open="previewVisible"
      title="表单预览"
      width="800px"
      :footer="null"
      @cancel="closePreview"
    >
      <div class="preview-container">
        <a-form :model="previewFormData" layout="vertical">
          <template v-for="(item, index) in formItems" :key="index">
            <a-form-item
              :label="item.label"
              :name="item.id"
              :rules="[{ required: item.props.required, message: `请${getPlaceholder(item)}` }]"
            >
              <template v-if="item.type === 'input'">
                <a-input
                  :value="previewFormData[item.id]"
                  @update:value="(val) => updateFormData(item.id, val)"
                  :placeholder="getPlaceholder(item)"
                  class="custom-input"
                />
              </template>
              
              <template v-else-if="item.type === 'textarea'">
                <a-textarea
                  :value="previewFormData[item.id]"
                  @update:value="(val) => updateFormData(item.id, val)"
                  :placeholder="getPlaceholder(item)"
                  class="custom-input"
                />
              </template>
              
              <template v-else-if="item.type === 'number'">
                <a-input-number
                  :value="previewFormData[item.id]"
                  @update:value="(val) => updateFormData(item.id, val)"
                  :placeholder="getPlaceholder(item)"
                  class="custom-input"
                  style="width: 100%"
                />
              </template>
              
              <template v-else-if="item.type === 'select'">
                <a-select
                  :value="previewFormData[item.id]"
                  @update:value="(val) => updateFormData(item.id, val)"
                  :placeholder="getPlaceholder(item)"
                  class="custom-input"
                  style="width: 100%"
                  v-bind="item.props"
                />
              </template>
              
              <template v-else>
                <component
                  :is="getPreviewComponent(item.type)"
                  :value="previewFormData[item.id]"
                  @update:value="(val) => updateFormData(item.id, val)"
                  v-bind="getPreviewProps(item)"
                  :placeholder="getPlaceholder(item)"
                  class="custom-input"
                >
                  <template v-if="item.type === 'upload'" #default>
                    <div v-if="item.props.listType === 'picture-card'">
                      <plus-outlined />
                      <div style="margin-top: 8px">点击上传</div>
                    </div>
                    <a-button v-else>
                      <upload-outlined />
                      点击上传
                    </a-button>
                  </template>
                </component>
              </template>
            </a-form-item>
          </template>
          <div class="preview-actions">
            <a-space>
              <a-button type="primary" @click="handlePreviewSubmit">提交</a-button>
              <a-button @click="handlePreviewReset">重置</a-button>
            </a-space>
          </div>
        </a-form>
      </div>
    </a-modal>
  </div>
</template>

<script>
import { PlusOutlined, UploadOutlined, ThunderboltOutlined } from '@ant-design/icons-vue'
import FormItemLibrary from '../components/FormDesigner/FormItemLibrary.vue'
import DesignArea from '../components/FormDesigner/DesignArea.vue'
import PropertyPanel from '../components/FormDesigner/PropertyPanel.vue'
import { message, Modal, Input } from 'ant-design-vue'
import { ref, reactive, h } from 'vue'
import axios from 'axios'

export default {
  name: 'FormDesigner',
  components: {
    FormItemLibrary,
    DesignArea,
    PropertyPanel,
    PlusOutlined,
    UploadOutlined,
    ThunderboltOutlined,
    ATextarea: Input.TextArea
  },
  data() {
    return {
      selectedItem: null,                // 当前选中的表单项
      previewVisible: false,           // 预览对话框是否可见
      previewFormData: reactive({}),   // 预览表单数据
      formItems: [],                    // 表单项列表
      $confirm: Modal.confirm,         // 确认对话框
      aiModalVisible: false,           // AI表单生成对话框是否可见
      aiPrompt: '',                    // AI表单生成提示
      aiGenerating: false,            // AI表单生成是否正在进行
      showDesignArea: true,            // 是否显示设计区域
      designAreaKey: 0,                 // 设计区域组件的唯一标识
    }
  },
  methods: {
    onItemSelected(item) {
      this.selectedItem = item
    },
    async saveForm() {
      try {
        if (!this.$refs.designArea) {
          message.error('无法获取表单配置');
          return;
        }

        // 获取表单项的深拷贝，避免引用问题
        const formItems = JSON.parse(JSON.stringify(this.$refs.designArea.formItems || []));
        
        if (formItems.length === 0) {
          message.warning('表单内容为空，请先添加表单项');
          return;
        }

        // 保存到后端
        const formData = {
          title: '新建表单',
          config: JSON.stringify({
            fields: formItems
          })
        };

        const response = await axios.post('http://localhost:8000/api/forms', formData);
        
        if (response.data) {
          // 同时保存到本地存储
          localStorage.setItem('saved_form_config', JSON.stringify({
            timestamp: new Date().toISOString(),
            items: formItems
          }));
          
          console.log('Form saved:', response.data);
          message.success('表单保存成功');
        } else {
          throw new Error('保存失败：服务器返回数据无效');
        }
      } catch (error) {
        console.error('Error saving form:', error);
        message.error('保存失败：' + (error.response?.data?.message || error.message));
      }
    },
    exportForm() {
      const formConfig = {
        items: this.$refs.designArea.formItems,
        timestamp: new Date().toISOString()
      }
      const blob = new Blob([JSON.stringify(formConfig, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `form_config_${new Date().getTime()}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    },
    importForm(file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        try {
          const config = JSON.parse(e.target.result)
          this.$refs.designArea.formItems = config.items
          message.success('表单已导入')
        } catch (error) {
          message.error('导入失败：无效的配置文件')
        }
      }
      reader.readAsText(file)
      return false
    },
    // 预览相关方法
    previewForm() {
      this.formItems = this.$refs.designArea.formItems
      if (this.formItems.length === 0) {
        message.warning('请先添加表单项')
        return
      }
      
      // 重置表单数据
      this.previewFormData = reactive({})
      
      // 为每个表单项创建独立的数据存储
      this.formItems.forEach(item => {
        this.previewFormData[item.id] = undefined
      })
      
      this.previewVisible = true
    },
    closePreview() {
      this.previewVisible = false
      this.previewFormData = reactive({})
    },
    getPreviewComponent(type) {
      const componentMap = {
        'input': 'a-input',
        'textarea': 'a-textarea',
        'select': 'a-select',
        'radio': 'a-radio-group',
        'checkbox': 'a-checkbox-group',
        'date': 'a-date-picker',
        'datetime': 'a-date-picker',
        'time': 'a-time-picker',
        'date-range': 'a-range-picker',
        'datetime-range': 'a-range-picker',
        'number': 'a-input-number',
        'switch': 'a-switch',
        'rate': 'a-rate',
        'slider': 'a-slider',
        'upload': 'a-upload'
      }
      return componentMap[type] || 'div'
    },
    getPreviewProps(item) {
      const props = { ...item.props }
      if (item.type === 'upload') {
        return {
          ...props,
          customRequest: ({ file, onSuccess }) => {
            setTimeout(() => {
              onSuccess('ok')
              message.success(`${file.name} 上传成功`)
            }, 1000)
          }
        }
      }
      
      // 为不同类型的组件添加占位符
      const placeholderText = this.getPlaceholder(item)
      switch (item.type) {
        case 'select':
          return { ...props, placeholder: placeholderText }
        case 'date':
        case 'datetime':
        case 'time':
        case 'date-range':
        case 'datetime-range':
          return { ...props, placeholder: placeholderText }
        case 'number':
          return { ...props, placeholder: placeholderText }
        default:
          return props
      }
    },
    getPlaceholder(item) {
      const actionMap = {
        'input': '输入',
        'textarea': '输入',
        'select': '选择',
        'radio': '选择',
        'checkbox': '选择',
        'date': '选择',
        'datetime': '选择',
        'time': '选择',
        'date-range': '选择',
        'datetime-range': '选择',
        'number': '输入数字',
        'rate': '选择',
        'slider': '选择',
        'upload': '上传'
      }
      return `请${actionMap[item.type] || '输入'}${item.label}`
    },
    handlePreviewSubmit() {
      // 获取当前时间戳作为唯一标识
      const timestamp = new Date().getTime()
      
      // 构建要保存的数据对象
      const formData = {
        id: timestamp,
        timestamp: new Date().toISOString(),
        formConfig: this.formItems,
        formData: this.previewFormData
      }
      
      // 从localStorage获取已有的数据
      const savedFormsStr = localStorage.getItem('submitted_forms') || '[]'
      const savedForms = JSON.parse(savedFormsStr)
      
      // 添加新的表单数据
      savedForms.push(formData)
      
      // 保存回localStorage
      localStorage.setItem('submitted_forms', JSON.stringify(savedForms))
      
      // 导出为JSON文件
      const blob = new Blob([JSON.stringify(formData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `form_data_${timestamp}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
      
      message.success('表单已保存并导出')
      
      // 关闭预览窗口
      this.closePreview()
    },
    handlePreviewReset() {
      this.previewFormData = reactive({})
      message.info('表单已重置')
    },
    // 确保每个表单组件获取独立的数据副本
    getInitialValue(field) {
      // 根据字段类型返回一个新的数据副本
      if (field.type === 'input') {
        return '';
      }
      // 为其他类型的字段添加相应的初始值
      return null;
    },
    // 在模板中使用时，修改 v-model 绑定
    getFieldKey(item) {
      return `${item.props.name}_${item.id}`
    },
    // 添加新方法用于更新表单数据
    updateFormData(id, value) {
      this.previewFormData[id] = value
    },
    // 添加清空所有表单的方法
    clearAllForms() {
      if (this.$refs.designArea) {
        try {
          // 移除选中状态
          this.selectedItem = null;
          
          // 直接修改数组长度为0，这样可以避免触发不必要的重渲染
          this.$refs.designArea.formItems.length = 0;
          
          // 触发更新
          this.$refs.designArea.$forceUpdate();
          
          message.success('已清空所有表单项');
        } catch (error) {
          console.error('清空表单时出错:', error);
          message.error('清空表单失败');
        }
      }
    },
    handleAddComponent(component) {
      // 调用 DesignArea 的添加组件方法
      this.$refs.designArea.addComponent(component)
    },
    async generateFormWithAI() {
      this.aiModalVisible = true;
    },
    /**
     * 处理AI表单生成
     * 该方法会调用后端API，根据用户输入的描述生成对应的表单配置
     */
    async handleAIFormGenerate() {
      // 验证用户输入不能为空
      if (!this.aiPrompt.trim()) {
        message.warning('请输入表单描述');
        return;
      }

      // 设置生成中状态，用于显示加载动画
      this.aiGenerating = true;
      // 获取设计区域组件引用
      const designArea = this.$refs.designArea;
      
      try {
        // 调用Node.js后端的AI生成接口
        const response = await axios.post('http://localhost:3000/api/generate-form', {
          prompt: this.aiPrompt // 发送用户输入的描述
        });

        // 如果成功生成表单配置
        if (response.data.success && response.data.data) {
          // 确保设计区域组件存在
          if (designArea) {
            // 清空现有表单项，准备添加新生成的组件
            designArea.formItems = [];
            
            // 遍历生成的组件配置，逐个添加到设计区域
            for (const component of response.data.data) {
              console.log('Adding component:', component);
              designArea.addComponent(component);
            }
            
            // 显示成功提示
            message.success('表单生成成功！');
          }
        } else {
          // 如果返回数据无效，显示警告信息
          message.warning(response.data.error || '未能识别具体的表单字段，请尝试更详细的描述');
        }
        
        // 关闭AI生成对话框
        this.aiModalVisible = false;
        // 清空输入框
        this.aiPrompt = '';
      } catch (error) {
        // 捕获并处理错误
        console.error('表单生成错误:', error);
        message.error('表单生成失败：' + error.message);
      } finally {
        // 无论成功与否，都需要关闭加载状态
        this.aiGenerating = false;
      }
    },
    /**
     * 处理AI表单生成对话框的取消操作
     * 关闭对话框并清空用户输入
     */
    handleAIFormCancel() {
      this.aiModalVisible = false; // 关闭对话框
      this.aiPrompt = ''; // 清空输入内容
    }
  },
  async mounted() {
    try {
      // 尝试从本地存储恢复
      const savedConfig = localStorage.getItem('saved_form_config');
      if (savedConfig) {
        const config = JSON.parse(savedConfig);
        if (config.items && Array.isArray(config.items)) {
          // 等待组件完全挂载
          await this.$nextTick();
          if (this.$refs.designArea) {
            this.$refs.designArea.formItems = config.items;
            message.success('已恢复上次保存的表单');
            return;
          }
        }
      }

      // 如果本地没有保存的配置，尝试从后端获取最新的表单
      const response = await axios.get('http://localhost:8000/api/forms/latest');
      if (response.data && response.data.config) {
        const config = JSON.parse(response.data.config);
        if (config.fields && Array.isArray(config.fields)) {
          await this.$nextTick();
          if (this.$refs.designArea) {
            this.$refs.designArea.formItems = config.fields;
            message.success('已加载最新保存的表单');
          }
        }
      }
    } catch (error) {
      console.error('Failed to load saved form config:', error);
      message.error('加载保存的表单失败');
    }
  }
}
</script>

<style scoped>
.form-designer {
  height: 100vh;
}

.header {
  padding: 0 24px;
  background: #fff;
  border-bottom: 1px solid #f0f0f0;
  height: 64px;
}

.header-content {
  display: flex;
  justify-content: space-between; /* 使logo和按钮组分别靠左和靠右 */
  align-items: center;
  height: 100%;
  width: 100%;
}

.logo {
  font-size: 18px;
  font-weight: bold;
}

.actions {
  display: flex;
  gap: 8px;
}

.content {
  padding: 24px;
  background: #fff;
}

.inner-layout {
  background: #fff;
  border: 1px solid #f0f0f0;
  border-radius: 4px;
  height: calc(100vh - 140px);
}

.sider {
  background: #fff;
}

.main-content {
  background: #fff;
  padding: 0;
}

.preview-container {
  padding: 24px;
  max-height: 70vh;
  overflow-y: auto;
}

.preview-actions {
  margin-top: 24px;
  text-align: center;
}

:deep(.ant-modal-body) {
  padding: 0;
}

:deep(.ant-upload-select) {
  width: 100%;
}

:deep(.ant-upload.ant-upload-select-picture-card),
:deep(.ant-upload.ant-upload-select-picture-circle) {
  width: 104px;
  height: 104px;
  margin-right: 8px;
  margin-bottom: 8px;
}

:deep(.ant-upload-list-picture-card .ant-upload-list-item),
:deep(.ant-upload-list-picture-circle .ant-upload-list-item) {
  width: 104px;
  height: 104px;
  margin: 0 8px 8px 0;
}

:deep(.ant-upload-list) {
  display: flex;
  flex-wrap: wrap;
}

/* 统一所有占位符的样式 */
:deep(.ant-input::placeholder),
:deep(.ant-textarea::placeholder),
:deep(.ant-input-number-input::placeholder),
:deep(.ant-select-selection-placeholder),
:deep(.ant-picker-input > input::placeholder) {
  color: rgba(0, 0, 0, 0.45) !important;
  opacity: 1 !important;
}

/* 统一所有获得焦点时的占位符样式 */
:deep(.ant-input:focus::placeholder),
:deep(.ant-textarea:focus::placeholder),
:deep(.ant-input-number-focused .ant-input-number-input::placeholder),
:deep(.ant-select-focused .ant-select-selection-placeholder),
:deep(.ant-picker-focused .ant-picker-input > input::placeholder) {
  opacity: 0 !important;
}
</style> 