package model

import (
	"blog/utils/errmsg"
	"context"
	"github.com/qiniu/api.v7/v7/auth/qbox"
	"github.com/qiniu/api.v7/v7/storage"
	"github.com/spf13/viper"
	"mime/multipart"
)

var AccessKey = viper.GetString("mini.AccessKey")
var SecretKey = viper.GetString("mini.SecretKey")
var Bucket = viper.GetString("mini.Bucket")
var ImgUrl = viper.GetString("mini.QinServer")

func UpLoadFile(file multipart.File, fileSize int64) (string, int) {
	putPolicy := storage.PutPolicy{
		//Scope: Bucket,
		Scope: "ggggblog",
	}

	//mac := qbox.NewMac(AccessKey, SecretKey)
	mac := qbox.NewMac("Xc7KD48OstUl9OwedKM5PRvi9W3f3Z2ZxUdvCYH7", "RXweIs1WQCV6F_22akiuoUlA9TLHgZsZENa4QI3P")
	upToken := putPolicy.UploadToken(mac)

	config := storage.Config{
		Zone:          &storage.ZoneHuanan,
		UseCdnDomains: false,
		UseHTTPS:      false,
	}

	putExtra := storage.PutExtra{}

	formUploader := storage.NewFormUploader(&config)

	ret := storage.PutRet{}

	err := formUploader.PutWithoutKey(context.Background(), &ret, upToken, file, fileSize, &putExtra)
	if err != nil {
		return "", errmsg.ERROR
	}
	//url := ImgUrl + ret.Key
	url := "http://sdplkhcej.hn-bkt.clouddn.com/" + ret.Key
	//url := "http://up-z2.qiniup.com/" + ret.Key
	return url, errmsg.SUCCESS
}
