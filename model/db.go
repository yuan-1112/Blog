package model

import (
	"fmt"
	"github.com/spf13/viper"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
	"time"
)

var db *gorm.DB

func InitDb() {
	logMode := logger.Info
	if !viper.GetBool("mode.develop") {
		logMode = logger.Error
	}
	var err error
	db, err = gorm.Open(mysql.Open(viper.GetString("db.dsn")), &gorm.Config{Logger: logger.Default.LogMode(logMode)})
	if err != nil {
		fmt.Println("数据库连接失败，请检查参数：", err)
		return
	}
	sqlDB, _ := db.DB()
	err = db.AutoMigrate(&User{}, &Article{}, &Category{})
	if err != nil {
		return
	}
	sqlDB.SetMaxIdleConns(viper.GetInt("db.SetMaxIdleConns"))
	sqlDB.SetMaxOpenConns(viper.GetInt("db.SetMaxOpenConns"))
	sqlDB.SetConnMaxLifetime(10 * time.Second)
}
