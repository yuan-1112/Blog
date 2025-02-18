<template>
  <div class="form-item-library">
    <h2>组件库</h2>
    <a-collapse v-model:activeKey="activeCategories" :bordered="false">
      <!-- 输入型组件 -->
      <a-collapse-panel key="1" header="输入型组件">
        <div class="components-list">
          <div v-for="item in inputComponents" :key="item.type + item.label"
            class="component-item" draggable="true" @dragstart="handleDragStart($event, item)" @click="handleComponentClick(item)">
            <a-card size="small">{{ item.label }}</a-card>
          </div>
        </div>
      </a-collapse-panel>

      <!-- 选择型组件 -->
      <a-collapse-panel key="2" header="选择型组件">
        <div class="components-list">
          <div v-for="item in selectComponents" :key="item.type + item.label"
            class="component-item" draggable="true" @dragstart="handleDragStart($event, item)" @click="handleComponentClick(item)">
            <a-card size="small">{{ item.label }}</a-card>
          </div>
        </div>
      </a-collapse-panel>

      <!-- 日期时间组件 -->
      <a-collapse-panel key="3" header="日期时间">
        <div class="components-list">
          <div v-for="item in dateComponents" :key="item.type + item.label"
            class="component-item" draggable="true" @dragstart="handleDragStart($event, item)" @click="handleComponentClick(item)">
            <a-card size="small">{{ item.label }}</a-card>
          </div>
        </div>
      </a-collapse-panel>

      <!-- 上传组件 -->
      <a-collapse-panel key="4" header="上传组件">
        <div class="components-list">
          <div v-for="item in uploadComponents" :key="item.type + item.label"
            class="component-item" draggable="true" @dragstart="handleDragStart($event, item)" @click="handleComponentClick(item)">
            <a-card size="small">{{ item.label }}</a-card>
          </div>
        </div>
      </a-collapse-panel>
    </a-collapse>
  </div>
</template>

<script>
export default {
  name: 'FormItemLibrary',
  data() {
    return {
      activeCategories: ['1', '2', '3', '4'],
      inputComponents: [
        { 
          type: 'input',
          label: '单行输入框',
          props: {
            placeholder: '请输入',
            allowClear: true,
            name: 'input'
          }
        },
        {
          type: 'textarea',
          label: '多行输入框',
          props: {
            placeholder: '请输入',
            autoSize: { minRows: 2, maxRows: 6 },
            allowClear: true,
            name: 'textarea'
          }
        },
        {
          type: 'input',
          label: '学号输入框',
          props: {
            placeholder: '请输入学号',
            pattern: '^[0-9]{8}$',
            maxLength: 8,
            name: 'studentid'
          }
        },
        {
          type: 'number',
          label: '数字输入框',
          props: {
            placeholder: '请输入数字',
            min: 0,
            max: 100,
            step: 1,
            name: 'number'
          }
        },
        {
          type: 'input',
          label: '密码输入框',
          props: {
            type: 'password',
            placeholder: '请输入密码',
            allowClear: true,
            name: 'password'
          }
        },
        {
          type: 'input',
          label: '邮箱输入框',
          props: {
            type: 'email',
            placeholder: '请输入邮箱地址',
            pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$',
            name: 'email'
          }
        },
        {
          type: 'input',
          label: '手机号输入框',
          props: {
            type: 'tel',
            placeholder: '请输入手机号',
            pattern: '^1[3-9]\\d{9}$',
            maxLength: 11,
            name: 'phone'
          }
        }
      ],
      selectComponents: [
        {
          type: 'select',
          label: '下拉选择框',
          props: {
            placeholder: '请选择',
            allowClear: true,
            options: [],
            name: 'select'
          }
        },
        {
          type: 'select',
          label: '多选下拉框',
          props: {
            mode: 'multiple',
            placeholder: '请选择',
            allowClear: true,
            options: [],
            name: 'multiselect'
          }
        },
        {
          type: 'radio',
          label: '单选框组',
          props: {
            options: [
              { label: '选项1', value: '1' },
              { label: '选项2', value: '2' }
            ],
            name: 'radio'
          }
        },
        {
          type: 'checkbox',
          label: '复选框组',
          props: {
            options: [
              { label: '选项1', value: '1' },
              { label: '选项2', value: '2' }
            ],
            name: 'checkbox'
          }
        },
        {
          type: 'switch',
          label: '开关',
          props: {
            checkedChildren: '开',
            unCheckedChildren: '关',
            name: 'switch'
          }
        },
        {
          type: 'rate',
          label: '评分',
          props: {
            allowHalf: true,
            count: 5,
            name: 'rate'
          }
        },
        {
          type: 'slider',
          label: '滑动输入条',
          props: {
            min: 0,
            max: 100,
            step: 1,
            name: 'slider'
          }
        }
      ],
      dateComponents: [
        {
          type: 'date',
          label: '日期选择',
          props: {
            placeholder: '请选择日期',
            format: 'YYYY-MM-DD',
            name: 'date'
          }
        },
        {
          type: 'datetime',
          label: '日期时间',
          props: {
            placeholder: '请选择日期时间',
            format: 'YYYY-MM-DD HH:mm:ss',
            name: 'datetime'
          }
        },
        {
          type: 'time',
          label: '时间选择',
          props: {
            placeholder: '请选择时间',
            format: 'HH:mm:ss',
            name: 'time'
          }
        },
        {
          type: 'date-range',
          label: '日期范围',
          props: {
            placeholder: ['开始日期', '结束日期'],
            format: 'YYYY-MM-DD',
            name: 'date_range'
          }
        },
        {
          type: 'datetime-range',
          label: '日期时间范围',
          props: {
            placeholder: ['开始时间', '结束时间'],
            format: 'YYYY-MM-DD HH:mm:ss',
            name: 'datetime_range'
          }
        }
      ],
      uploadComponents: [
        {
          type: 'upload',
          label: '文件上传',
          props: {
            action: '',
            multiple: true,
            accept: '*/*',
            name: 'fileupload'
          }
        },
        {
          type: 'upload',
          label: '图片上传',
          props: {
            action: '',
            listType: 'picture-card',
            accept: 'image/*',
            multiple: true,
            name: 'image_upload'
          }
        },
        {
          type: 'upload',
          label: '头像上传',
          props: {
            action: '',
            listType: 'picture-card',
            accept: 'image/*',
            maxCount: 1,
            showUploadList: false,
            name: 'avatar_upload'
          }
        }
      ]
    }
  },
  methods: {
    handleDragStart(event, item) {
      try {
        // 创建组件数据的深拷贝，避免引用问题
        const componentData = JSON.parse(JSON.stringify(item));
        
        // 确保数据格式正确，至少含有字段名和标签
        if (!componentData.type || !componentData.label) {
          console.warn('Invalid component data structure');
          return;
        }
        
        // 设置拖拽数据
        event.dataTransfer.effectAllowed = 'copy';
        event.dataTransfer.setData('component', JSON.stringify(componentData));
      } catch (error) {
        console.error('Drag start error:', error);
        message.error('拖拽初始化失败');
      }
    },
    handleComponentClick(item) {
      this.$emit('add-component', item);
    }
  }
}
</script>

<style>
.form-item-library {
  width: 250px;
  border-right: 1px solid #eee;
  padding: 16px;
}

.components-list {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin: 8px 0;
}

.component-item {
  cursor: grab;
}

.component-item:active {
  cursor: grabbing;
}

.component-item:hover {
  opacity: 0.8;
}

:deep(.ant-card-small) {
  border: 1px solid #e8e8e8;
}

:deep(.ant-card-small:hover) {
  border-color: #1890ff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
}

:deep(.ant-collapse) {
  background: transparent;
}

:deep(.ant-collapse-header) {
  font-weight: bold;
}
</style> 