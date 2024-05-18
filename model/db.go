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
	//AutoMigrate 设置自动迁移
	err = db.AutoMigrate(&User{}, &Article{}, &Category{})
	if err != nil {
		return
	}

	//SetConnMaxIdleTime 设置连接池中的最大闲置连接数
	sqlDB.SetMaxIdleConns(viper.GetInt("db.SetMaxIdleConns"))

	//SetMaxOpenConns 设置连接池中的最大连接数量
	sqlDB.SetMaxOpenConns(viper.GetInt("db.SetMaxOpenConns"))

	//SetConnMaxLifetime 设置连接的最大可复用时间
	sqlDB.SetConnMaxLifetime(10 * time.Second)

	//db.Close()

}
