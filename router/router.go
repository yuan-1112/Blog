package router

import (
	"blog/api/v1"
	"blog/middleware"
	"github.com/gin-gonic/gin"
)

func InitRouter() {

	r := gin.Default()

	//公共路由
	rgPublic := r.Group("api/v1/public")
	{
		//用户模块的路由接口
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

	//鉴权路由
	rgAuth := r.Group("api/v1")
	rgAuth.Use(middleware.JwtToken())
	{
		//用户模块的路由接口
		user := rgAuth.Group("user")
		{
			//user.POST("add", v1.AddUser)
			//user.GET("list", v1.GetUsers)
			user.PUT(":id", v1.EditUser)
			user.DELETE(":id", v1.DeleteUser)
		}
		//分类模块的路由接口
		category := rgAuth.Group("category")
		{
			category.POST("add", v1.AddCategory)
			//category.GET("list", v1.GetCategory)
			category.PUT(":id", v1.EditCategory)
			category.DELETE(":id", v1.DeleteCategory)
		}
		//文章模块的路由接口
		article := rgAuth.Group("article")
		{
			article.POST("add", v1.AddArticle)
			//article.GET("list", v1.GetArticle)
			article.PUT(":id", v1.EditArticle)
			article.DELETE(":id", v1.DeleteArticle)
			//article.GET("info/:id", v1.GetArticleInfo)
			//article.GET("list/:id", v1.GetCategoryArticle)
		}

	}

	//port := viper.GetString("server.port")
	r.Run(":9090")
}
