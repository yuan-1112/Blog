<template>
  <div class="property-panel">
    <h2>属性面板</h2>
    <div v-if="selectedItem" class="properties-container">
      <a-form layout="vertical">
        <!-- 基础属性 -->
        <a-form-item label="字段名">
          <a-input
            v-model:value="selectedItem.props.name"
            :placeholder="getDefaultFieldName(selectedItem)"
            @focus="handleFieldNameFocus"
            @blur="handleFieldNameBlur"
          />
        </a-form-item>
        
        <a-form-item label="标签">
          <a-input
            v-model:value="selectedItem.label"
            :placeholder="selectedItem.label"
            @focus="handleLabelFocus"
            @blur="handleLabelBlur"
          />
        </a-form-item>

        <!-- 验证规则 -->
        <a-divider>验证规则</a-divider>
        
        <a-form-item>
          <a-checkbox v-model:checked="selectedItem.props.required">必填</a-checkbox>
        </a-form-item>

        <template v-if="selectedItem.type === 'input'">
          <a-form-item label="最小长度">
            <a-input-number v-model:value="selectedItem.props.minLength" :min="0" />
          </a-form-item>
          
          <a-form-item label="最大长度">
            <a-input-number v-model:value="selectedItem.props.maxLength" :min="0" />
          </a-form-item>
        </template>

        <template v-if="selectedItem.type === 'number'">
          <a-form-item label="最小值">
            <a-input-number v-model:value="selectedItem.props.min" />
          </a-form-item>
          
          <a-form-item label="最大值">
            <a-input-number v-model:value="selectedItem.props.max" />
          </a-form-item>
        </template>

        <!-- 选项配置 -->
        <template v-if="['select', 'radio', 'checkbox'].includes(selectedItem.type)">
          <a-divider>选项配置</a-divider>
          
          <div v-for="(option, index) in selectedItem.props.options" :key="index" class="option-item">
            <a-input v-model:value="option.label" placeholder="选项文本" />
            <a-input v-model:value="option.value" placeholder="选项值" />
            <a-button type="text" @click="removeOption(index)">
              <delete-outlined />
            </a-button>
          </div>
          
          <a-button type="dashed" block @click="addOption">
            <plus-outlined />添加选项
          </a-button>
        </template>
      </a-form>
    </div>
    <div v-else class="empty-tip">
      请选择一个表单项进行编辑
    </div>
  </div>
</template>

<script>
import { DeleteOutlined, PlusOutlined } from '@ant-design/icons-vue'

export default {
  name: 'PropertyPanel',
  components: {
    DeleteOutlined,
    PlusOutlined
  },
  props: {
    selectedItem: {
      type: Object,
      default: null
    }
  },
  data() {
    return {
      typeCounters: {} // 用于记录每种类型的计数
    }
  },
  methods: {
    addOption() {
      if (!this.selectedItem.props.options) {
        this.selectedItem.props.options = []
      }
      this.selectedItem.props.options.push({
        label: '',
        value: ''
      })
    },
    removeOption(index) {
      this.selectedItem.props.options.splice(index, 1)
    },
    // 获取默认字段名
    getDefaultFieldName(item) {
      if (!item) return ''
      
      // 如果已经有名字，就返回现有的
      if (item.props.name && String(item.props.name).trim() !== '') {
        return String(item.props.name)
      }
      
      // 直接返回类型名称作为字段名
      const typeMap = {
        'input': 'input',
        'textarea': 'textarea',
        'select': 'select',
        'radio': 'radio',
        'checkbox': 'checkbox',
        'date': 'date',
        'datetime': 'datetime',
        'time': 'time',
        'date-range': 'daterange',
        'datetime-range': 'datetimerange',
        'number': 'number',
        'switch': 'switch',
        'rate': 'rate',
        'slider': 'slider',
        'upload': 'upload'
      }
      
      return typeMap[item.type] || item.type
    },

    // 处理字段名输入框的焦点事件
    handleFieldNameFocus(e) {
      const currentName = String(this.selectedItem.props.name || '');
      if (!currentName || currentName === this.getDefaultFieldName(this.selectedItem)) {
        this.selectedItem.props.name = ''
      }
    },

    handleFieldNameBlur(e) {
      const currentName = String(this.selectedItem.props.name || '');
      if (!currentName || currentName.trim() === '') {
        this.selectedItem.props.name = this.getDefaultFieldName(this.selectedItem)
      }
    },

    // 处理标签输入框的焦点事件
    handleLabelFocus(e) {
      const currentLabel = String(this.selectedItem.label || '');
      if (!currentLabel || currentLabel === this.selectedItem.defaultLabel) {
        this.selectedItem.label = ''
      }
    },

    handleLabelBlur(e) {
      const currentLabel = String(this.selectedItem.label || '');
      if (!currentLabel || currentLabel.trim() === '') {
        this.selectedItem.label = this.selectedItem.defaultLabel
      }
    }
  },
  mounted() {
    // 初始化计数器
    if (this.selectedItem) {
      this.selectedItem.props.name = this.getDefaultFieldName(this.selectedItem)
    }
  },
  watch: {
    selectedItem: {
      immediate: true,
      handler(newItem) {
        if (newItem && !newItem.props.name) {
          newItem.props.name = this.getDefaultFieldName(newItem)
        }
      }
    }
  }
}
</script>

<style>
.property-panel {
  width: 300px;
  border-left: 1px solid #eee;
  padding: 16px;
}

.properties-container {
  margin-top: 16px;
}

.option-item {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  align-items: center;
}

.empty-tip {
  text-align: center;
  color: #999;
  padding: 32px;
}

:deep(.ant-input::placeholder) {
  color: rgba(0, 0, 0, 0.45) !important;
  opacity: 1 !important;
}

:deep(.ant-input:focus::placeholder) {
  opacity: 0 !important;
}
</style> 