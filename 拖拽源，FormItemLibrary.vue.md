表单生成器的Draggable讲解

### 拖拽源，FormItemLibrary.vue

### html部分

以输入型组件为例，

`<a-collapse>`是一个折叠面板组件，其中的组件以卡片形式进行展示，

- `v-model:activeKey="activeCategories"`: 这个属性控制哪个面板是展开的，进行了双向绑定，`activeCategories` 存储着面板 `key` 的数组。如果 `key` 为 "1" 的面板被点击展开，`activeCategories` 数组会自动包含 `"1"`，这样面板就会保持展开状态。



`<a-collapse-panel>`在一个折叠面板容器 (`a-collapse`) 中定义每个具体的面板。

- `key` 标识折叠面板
- `draggable="true"`将每个组件项设置为可拖拽
- `@dragstart="handleDragStart($event, item)"`: 当拖拽开始时，调用 `handleDragStart` 方法。
- `@click="handleComponentClick(item)"`: 组件项被点击时调用 `handleComponentClick` 方法，通常是用来处理选中组件的逻辑

### js部分

####  **handleDragStart(*event*, *item*)**

`event` 代表的是浏览器处理拖拽操作的过程中触发的事件。

`item`是被点击的组件对象。

**`const componentData = JSON.parse(JSON.stringify(item));`**

-  `JSON.parse(JSON.stringify(item));`将item对象先转化为字符串，再转化为一个新的对象，主要是为了实现深拷贝，即新对象和之前对象的内存地址不同，完全独立，避免在后续操作中对原始数据的意外修改。

 **`event.dataTransfer.effectAllowed = 'copy';`**

- `event.dataTransfer` 是 **HTML5 拖拽 API** 中的核心对象，提供了对拖拽数据的操作能力

- `effectAllowed`指定拖拽时允许的操作类型

  

- | **值**     | **含义**                                   | **鼠标光标效果** |
  | ---------- | ------------------------------------------ | ---------------- |
  | `none`     | 不允许任何拖拽操作                         | 🚫 禁止符         |
  | `copy`     | 拖拽时表示要**复制**数据                   | ➕ 加号           |
  | `move`     | 拖拽时表示要**移动**数据                   | ↔️ 移动箭头       |
  | `link`     | 拖拽时表示要**创建链接**                   | 🔗 链接图标       |
  | `copyMove` | **允许复制和移动**（由目标区域决定）       | ➕/↔️              |
  | `copyLink` | **允许复制和链接**                         | ➕/🔗              |
  | `linkMove` | **允许链接和移动**                         | 🔗/↔️              |
  | `all`      | **允许所有操作**（`copy`、`move`、`link`） | 🔗/➕/↔️            |



**`event.dataTransfer.setData('component',JSON.stringify(componentData));`**

- `setData`用于在拖拽操作中将数据存储到 **拖拽事件** 中，以便在 `drop` 事件中通过 **`getData()`** 提取，他接受两个参数，一个是数据的格式，另外一个是要保存的数据，必须是字符串。

- `component` 这个数据格式是通过数据结构隐式定义的，它的结构在data() 函数中。

- `setData()` 只能接收字符串，复杂对象要用 `JSON.stringify()`。

  

#### handleComponentClick(item)

定义点击事件的处理函数，接收被点击的 `item`。

**`this.$emit()`**：**子组件向父组件**发送自定义事件。

- **`'add-component'`**：事件名称。
- **`item`**：点击时传递的组件数据。



### DesignArea.vue

### 拖拽目标 

#### html部分

- `@dragover.prevent`：用于阻止拖拽操作的默认行为。默认情况下，拖拽的内容不会被接受，必须阻止默认行为才能允许在目标区域放置拖拽的内容。
- `@drop="handleDrop"`：在拖拽的内容释放（drop）时，会触发 `handleDrop` 方法，这是处理拖拽数据的逻辑。

#### js部分

**handleDrop（）方法**

- `event.dataTransfer.getData('component')`：从拖拽事件中获取所拖拽的组件数据。`component` 是在拖拽源处设置的类型，表明拖拽的内容是一个组件。
- `this.addComponent(componentData)`：将拖拽进来的组件数据添加到设计区域的表单项列表中。

**addComponent（）方法**

- 将新的组件添加到表单项列表中，并设置一个唯一的 `id`。
- `this.formItems.push(newComponent)` 将新组件加入到表单项列表 `formItems` 中。
- `this.selectItem(newComponent)`：自动选中刚添加的组件。



#### 交换设计区域的元素

`v-model="formItems"`：`draggable` 组件的 `v-model` 用于双向绑定，`formItems` 用于存储拖拽的组件数据。

`group="form-items"`：指定拖拽组，确保组件在同一组内可以进行拖拽操作。

`item-key="id"`：指定每个拖拽项的唯一标识，`id` 在 `formItems` 中为每个组件设置的唯一值。

`ghost-class="ghost"`：当拖拽时，设置拖拽项的样式为 `ghost`，使其显示透明并有一点点漂浮感。
