package main

import (
	"blog/config"
	"blog/model"
	"blog/router"
)

func main() {
	initBase()
	//fmt.Println("Configurations:")
	//for key, value := range viper.AllSettings() {
	//	fmt.Printf("%s: %v\n", key, value)
	//}
}

func initBase() {
	config.InitConfig()
	model.InitDb()
	router.InitRouter()
}
