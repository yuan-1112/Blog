基于gin-web框架开发的一个博客项目

Day1 ：
    1.项目中创建api层、config层、middleware层、model层、router层、upload层、utils层 
        api：连接客户端和服务层，负责处理HTTP请求和返回响应，同时也负责处理身份验证和权限控制。 
        config：管理配置文件和提供配置参数 middleware：对HTTP请求进行预处理和后处理，实现通用功能和逻辑，提高代码的复用性和可维护性 
        model：负责数据结构的定义和数据访问逻辑的实现，帮助实现数据的持久化和管理。通常与数据库直接交互，为其他层提供数据操作接口。 
        router：扮演着路由分发和请求处理的角色，负责将HTTP请求映射到相应的处理函数上，并处理请求过程中的一些通用逻辑 
        upload：处理文件上传相关的逻辑，帮助实现文件上传功能。通常与Router层或Service层进行交互，处理文件上传请求并将文件信息传递给其他模块进行进一步处理。 
        utils：提供通用功能和工具函数，一个辅助性的模块。 
    2.用.yml文件写出配置文件，并在config层对配置文件设置初始化 
    3.在model层初始化Article、Category、User结构体，初始化数据库以及接着实现简单router测试

Day2:
    1.utils层写出errmsg，用于处理项目模块的各种情况需要捕捉的错误请求（也可以边写边加） 
    2.api层编写各类业务功能实现，并更新于router中的业务路由 
    3.model层完成gorm数据库操作，实现user、category的简单路由接口

Day3: 
    1.使用ScryptPw写出密码加密功能，用于数据库防护用户密码 
    2.完成article路由接口，查询分类下的所有文章、单个文章（article结构体涉及到category结构体，gorm操作需使用preload预加载）... 
    3.优化router层路由接口路径，分为user、article、category三组路径，登录时颁发token将其分为鉴权路由以及公共路由
    
Day4:
    1.添加博客文件上传功能，使用qiniu云服务器使其上传文件保存至服务器上
    2.编写jwtToken、cors跨域、logger日志处理中间件函数
    3.在添加用户功能模块中添加validator函数判断其合理化    
