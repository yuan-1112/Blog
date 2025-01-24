package main

import (
	"blog/config"
	"blog/model"
	"blog/router"
)

func main() {
	initBase()
}

func initBase() {
	config.InitConfig()
	model.InitDb()
	router.InitRouter()
}
