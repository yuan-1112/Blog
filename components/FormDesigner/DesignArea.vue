<template>
  <div class="design-area">
    <h2>设计区域</h2>
    <div
      class="design-container"
      @dragover.prevent
      @drop="handleDrop"
    >
      <div v-show="formItems.length === 0" class="empty-tip">
        拖拽组件到这里
      </div>
      <draggable
        v-model="formItems"
        group="form-items"
        item-key="id"
        class="form-container"
        ghost-class="ghost"
        @change="onChange"
      >
        <template #item="{ element }">
          <div
            class="form-item"
            :class="{ 'selected': selectedItem?.id === element.id }"
            @click="selectItem(element)"
            @contextmenu.prevent="showContextMenu($event, element)"
          >
            <div class="form-item-content">
              <div class="form-item-label">{{ element.label }}</div>
              <div class="form-item-field">
                <component
                  :is="getComponentType(element.type)"
                  v-bind="getComponentProps(element)"
                  class="form-component"
                  @change="handleUploadChange($event, element)"
                >
                  <template v-if="element.type === 'upload' && element.props.listType === 'picture-card'" #uploadButton>
                    <div>
                      <plus-outlined />
                      <div style="margin-top: 8px">上传</div>
                    </div>
                  </template>
                </component>
              </div>
            </div>
          </div>
        </template>
      </draggable>
    </div>

    <!-- 右键菜单 -->
    <a-dropdown :open="contextMenuVisible" :trigger="['contextmenu']" @openChange="handleContextMenuVisibleChange">
      <template #overlay>
        <a-menu @click="handleContextMenuClick">
          <a-menu-item key="delete">
            <delete-outlined />
            <span>删除</span>
          </a-menu-item>
        </a-menu>
      </template>
      <div class="context-menu-trigger" :style="contextMenuStyle"></div>
    </a-dropdown>
  </div>
</template>

<script>
import { DeleteOutlined, PlusOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import draggable from 'vuedraggable'

export default {
  name: 'DesignArea',
  components: {
    draggable,
    DeleteOutlined,
    PlusOutlined
  },
  data() {
    return {
      formItems: [],
      selectedItem: null,
      idCounter: 0,
      // 右键菜单相关
      contextMenuVisible: false,
      contextMenuPosition: { x: 0, y: 0 },
      contextMenuTarget: null,
      // 上传相关的数据
      fileList: new Map(),
      observer: null
    }
  },
  computed: {
    contextMenuStyle() {
      return {
        position: 'fixed',
        left: this.contextMenuPosition.x + 'px',
        top: this.contextMenuPosition.y + 'px',
        zIndex: 1000
      }
    }
  },
  methods: {
    handleDrop(event) {
      try {
        const data = event.dataTransfer.getData('component');
        if (!data) {
          console.warn('No data received in drop event');
          return;
        }
        
        let componentData;
        try {
          componentData = JSON.parse(data);
        } catch (parseError) {
          console.error('Failed to parse component data:', parseError);
          message.error('组件数据格式错误');
          return;
        }
        
        if (!componentData || typeof componentData !== 'object') {
          console.warn('Invalid component data format');
          return;
        }
        
        this.addComponent(componentData);
      } catch (error) {
        console.error('Drop handling error:', error);
        message.error('添加组件失败');
      }
    },
    addComponent(componentData) {
      // 确保组件数据有效
      if (!componentData || !componentData.type) {
        console.error('Invalid component data:', componentData);
        return;
      }

      console.log('Adding component with data:', componentData);

      try {
        // 生成唯一ID
        const newComponent = {
          id: `${componentData.type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          type: componentData.type,
          label: componentData.label || '未命名字段',
          props: {
            name: componentData.props?.name || `field_${Date.now()}`,
            placeholder: componentData.props?.placeholder || `请输入${componentData.label || '未命名字段'}`,
            required: Boolean(componentData.props?.required),
            ...componentData.props
          }
        };

        // 根据类型设置特定属性
        switch (newComponent.type) {
          case 'email':
            newComponent.props.type = 'email';
            break;
          case 'password':
            newComponent.props.type = 'password';
            break;
          case 'textarea':
            newComponent.props.rows = newComponent.props.rows || 4;
            break;
        }

        console.log('Created new component:', newComponent);

        // 添加到表单项列表
        this.formItems.push(newComponent);
        
        // 触发变更事件
        this.$emit('items-changed', this.formItems);
        
        // 自动选中新添加的组件
        this.selectItem(newComponent);

        return newComponent;
      } catch (error) {
        console.error('Error adding component:', error);
        throw error;
      }
    },
    getComponentType(type) {
      // 返回对应的组件类型
      const typeMap = {
        'input': 'a-input',
        'password': 'a-input-password',
        'email': 'a-input',
        'textarea': 'a-textarea',
        'number': 'a-input-number',
        'select': 'a-select',
        'radio': 'a-radio-group',
        'checkbox': 'a-checkbox-group',
        'date': 'a-date-picker',
        'time': 'a-time-picker',
        'switch': 'a-switch',
        'upload': 'a-upload'
      }
      return typeMap[type] || 'a-input'
    },
    getComponentProps(item) {
      // 基础属性
      const baseProps = {
        placeholder: item.props?.placeholder || `请输入${item.label}`,
        ...item.props
      }
      
      // 根据类型添加特定属性
      switch(item.type) {
        case 'email':
          return { ...baseProps, type: 'email' }
        case 'password':
          return { ...baseProps, type: 'password' }
        case 'textarea':
          return { ...baseProps, rows: 4 }
        case 'number':
          return { ...baseProps, style: { width: '100%' } }
        default:
          return baseProps
      }
    },
    getComponent(type) {
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
    getComponentProps(element) {
      if (element.type === 'upload') {
        return {
          ...element.props,
          fileList: this.fileList.get(element.id) || [],
          beforeUpload: (file) => this.handleBeforeUpload(file, element),
          customRequest: ({ file, onSuccess }) => this.customUploadRequest(file, onSuccess, element)
        }
      }
      return element.props
    },
    handleBeforeUpload(file, element) {
      const isImage = element.props.accept === 'image/*'
      if (isImage) {
        const isValidImage = file.type.startsWith('image/')
        if (!isValidImage) {
          message.error('只能上传图片文件！')
          return false
        }
        const isLt2M = file.size / 1024 / 1024 < 2
        if (!isLt2M) {
          message.error('图片必须小于 2MB！')
          return false
        }
      }
      return true
    },
    customUploadRequest(file, onSuccess, element) {
      // 这里模拟上传过程
      setTimeout(() => {
        const fileUrl = URL.createObjectURL(file)
        const fileItem = {
          uid: `${Date.now()}-${Math.random()}`,
          name: file.name,
          status: 'done',
          url: fileUrl,
          thumbUrl: element.props.accept === 'image/*' ? fileUrl : undefined
        }
        
        const currentFiles = this.fileList.get(element.id) || []
        const maxCount = element.props.maxCount || Infinity
        
        if (currentFiles.length >= maxCount) {
          currentFiles.shift() // 移除最旧的文件
        }
        
        const newFileList = [...currentFiles, fileItem]
        this.fileList.set(element.id, newFileList)
        
        onSuccess(fileItem)
        message.success(`${file.name} 上传成功！`)
      }, 1000)
    },
    handleUploadChange({ fileList }, element) {
      this.fileList.set(element.id, fileList)
    },
    onChange() {
      this.$emit('items-changed', this.formItems)
    },
    // 右键菜单相关方法
    showContextMenu(event, item) {
      this.contextMenuPosition = {
        x: event.clientX,
        y: event.clientY
      }
      this.contextMenuTarget = item
      this.contextMenuVisible = true
      event.preventDefault()
    },
    handleContextMenuVisibleChange(visible) {
      if (!visible) {
        this.contextMenuVisible = false
        this.contextMenuTarget = null
      }
    },
    handleContextMenuClick({ key }) {
      if (key === 'delete' && this.contextMenuTarget) {
        this.deleteItem(this.contextMenuTarget)
      }
      this.contextMenuVisible = false
    },
    deleteItem(item) {
      const index = this.formItems.findIndex(i => i.id === item.id)
      if (index > -1) {
        this.formItems.splice(index, 1)
        if (this.selectedItem?.id === item.id) {
          this.selectedItem = null
          this.$emit('item-selected', null)
        }
      }
    },
    selectItem(item) {
      this.selectedItem = item
      this.$emit('item-selected', item)
    },
    // Add new clear method
    clearForm() {
      // 清空选中状态
      this.selectedItem = null;
      this.$emit('item-selected', null);
      
      // 清空表单项
      this.formItems = [];
      
      // 清空文件列表
      this.fileList.clear();
      
      // 触发变更事件
      this.$emit('items-changed', this.formItems);
    }
  },
  mounted() {
    // 使用 ResizeObserver 的安全包装
    try {
      this.observer = new ResizeObserver(entries => {
        // 不执行任何操作，仅用于防止错误
      });
      this.observer.observe(this.$el);
    } catch (error) {
      console.warn('ResizeObserver not supported:', error);
    }
  },
  beforeUnmount() {
    // 清理 ResizeObserver
    if (this.observer) {
      this.observer.disconnect();
    }
  }
}
</script>

<style scoped>
.design-area {
  height: 100%;
  padding: 16px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.design-container {
  flex: 1;
  min-height: 200px;
  border: 2px dashed #d9d9d9;
  border-radius: 4px;
  padding: 16px;
  overflow-y: auto;
  position: relative;
}

.form-container {
  min-height: 100%;
  padding-bottom: 100px;
}

.form-item {
  position: relative;
  padding: 8px;
  margin-bottom: 16px;
  border: 1px solid transparent;
  border-radius: 4px;
  background: #fff;
  transition: all 0.3s;
}

.form-item:hover {
  border-color: #1890ff;
}

.form-item.selected {
  border-color: #1890ff;
  background-color: #e6f7ff;
}

.empty-tip {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #999;
  font-size: 14px;
  pointer-events: none;
}

/* 自定义滚动条样式 */
.design-container::-webkit-scrollbar {
  width: 6px;
}

.design-container::-webkit-scrollbar-thumb {
  background-color: #ccc;
  border-radius: 3px;
}

.design-container::-webkit-scrollbar-track {
  background-color: #f5f5f5;
}

.form-item-content {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  width: 100%;
}

.form-item-label {
  width: 100px;
  text-align: left;
  color: rgba(0, 0, 0, 0.85);
  padding-top: 5px;
}

.form-item-field {
  flex: 1;
  max-width: calc(100% - 120px);
}

.form-component {
  width: 100%;
}

.ghost {
  opacity: 0.5;
  background: #c8ebfb;
}

.context-menu-trigger {
  position: fixed;
  width: 1px;
  height: 1px;
  background: transparent;
}

:deep(.ant-dropdown-menu-item) {
  display: flex;
  align-items: center;
  gap: 8px;
}

:deep(.ant-upload-select) {
  width: 100%;
}

:deep(.ant-upload-list-picture-card) {
  width: 100%;
}

:deep(.ant-upload.ant-upload-select-picture-card) {
  width: 104px;
  height: 104px;
  margin: 0;
}

:deep(.ant-upload-list-picture-card .ant-upload-list-item) {
  width: 104px;
  height: 104px;
}
</style> 