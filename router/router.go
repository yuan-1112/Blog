package router

import (
	"blog/api/v1"
	"blog/middleware"
	"github.com/gin-gonic/gin"
)

func InitRouter() {
	r := gin.New()
	r.Use(middleware.Logger())
	r.Use(gin.Recovery())
	r.Use(middleware.Cors())
	rgPublic := r.Group("api/v1/public")
	{
		user := rgPublic.Group("user")
		{
			user.POST("add", v1.AddUser)
			user.GET("list", v1.GetUsers)
			user.POST("login", v1.Login)
		}
		category := rgPublic.Group("category")
		{
			category.GET("list", v1.GetCategory)
		}
		article := rgPublic.Group("article")
		{
			article.GET("list", v1.GetArticle)
			article.GET("info/:id", v1.GetArticleInfo)
			article.GET("list/:id", v1.GetCategoryArticle)
		}
	}
	rgAuth := r.Group("api/v1")
	rgAuth.Use(middleware.JwtToken())
	{
		user := rgAuth.Group("user")
		{
			user.POST("upload", v1.Upload) //上传文件接口
			user.PUT(":id", v1.EditUser)
			user.DELETE(":id", v1.DeleteUser)
		}
		category := rgAuth.Group("category")
		{
			category.POST("add", v1.AddCategory)
			category.PUT(":id", v1.EditCategory)
			category.DELETE(":id", v1.DeleteCategory)
		}
		article := rgAuth.Group("article")
		{
			article.POST("add", v1.AddArticle)
			article.PUT(":id", v1.EditArticle)
			article.DELETE(":id", v1.DeleteArticle)
		}
	}
	_ = r.Run(":9090")
}
