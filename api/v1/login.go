package v1

import (
	"blog/middleware"
	"blog/model"
	"blog/utils/errmsg"
	"github.com/gin-gonic/gin"
	"net/http"
)

func Login(c *gin.Context) {
	var data model.User
	_ = c.ShouldBindJSON(&data)
	var code int
	var token string
	code = model.CheckLogin(data.Username, data.Password)
	if code == errmsg.SUCCESS {
		//生成token
		token, code := middleware.SetToken(data.Username)
		c.JSON(http.StatusOK, gin.H{
			"status":  code,
			"message": errmsg.GetErrorMsg(code),
			"token":   token,
		})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"status":  code,
		"message": errmsg.GetErrorMsg(code),
		"token":   token,
	})
}
