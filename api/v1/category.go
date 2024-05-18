package v1

import (
	"blog/model"
	"blog/utils/errmsg"
	"github.com/gin-gonic/gin"
	"net/http"
	"strconv"
)

//查询分类是否存在

//添加分类

//查询单个分类下的文章

//查询文章列表

//编辑分类

//删除分类

func AddCategory(c *gin.Context) {
	var data model.Category
	_ = c.ShouldBindJSON(&data)

	code := model.CheckCategory(data.Name)
	if code == errmsg.SUCCESS {
		model.CreateCategory(&data)
	}
	if code == errmsg.ERROR_CATEGORY_USED {
		c.JSON(http.StatusConflict, gin.H{
			"error": "分类名已被使用",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  code,
		"data":    data,
		"message": errmsg.GetErrorMsg(code),
	})
}

func GetCategory(c *gin.Context) {
	pageSize, _ := strconv.Atoi(c.Query("pagesize"))
	pageNum, _ := strconv.Atoi(c.Query("pagenum"))

	if pageSize == 0 {
		//取小传输限制
		pageSize = -1
	}
	if pageNum == 0 {
		pageNum = -1
	}

	data := model.GetCategory(pageSize, pageNum)
	code := errmsg.SUCCESS
	c.JSON(http.StatusOK, gin.H{
		"status":  code,
		"data":    data,
		"message": errmsg.GetErrorMsg(code),
	})
}

func EditCategory(c *gin.Context) {
	//拿到ID，以及需要更新的内容
	var data model.Category
	_ = c.ShouldBindJSON(&data)
	code := model.CheckCategory(data.Name)
	id, _ := strconv.Atoi(c.Param("id"))
	if code == errmsg.SUCCESS {
		model.EditCategory(id, &data)
	}
	if code == errmsg.ERROR_CATEGORY_USED {
		c.Abort()
	}

	c.JSON(http.StatusOK, gin.H{
		"status":  code,
		"message": errmsg.GetErrorMsg(code),
	})

}

func DeleteCategory(c *gin.Context) {
	id, _ := strconv.Atoi(c.Param("id"))

	code := model.DeleteCategory(id)

	c.JSON(http.StatusOK, gin.H{
		"status":  code,
		"message": errmsg.GetErrorMsg(code),
	})
}
