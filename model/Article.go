package model

import (
	"blog/utils/errmsg"
	"github.com/jinzhu/gorm"
)

type Article struct {
	gorm.Model
	CategoryID uint
	Category   Category `gorm:"foreignkey:Cid"`
	Title      string   `gorm:"type: varchar(100); not null" json:"title"`
	Cid        int      `gorm:"type: int; not null" json:"cid"`
	Desc       string   `gorm:"type: varchar(200); not null" json:"desc"`
	Content    string   `gorm:"type: longtext;" json:"content"`
	Img        string   `gorm:"type: varchar(100);" json:"img"`
}

func CreateArticle(data *Article) int {
	err := db.Create(&data).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}

// GetCategoryArticle 查询分类下的所有文章
func GetCategoryArticle(id int, pageNum int, pageSize int) ([]Article, int, int64) {
	var cateArticleList []Article
	var total int64
	err := db.Preload("Category").Limit(pageSize).Offset((pageNum-1)*pageSize).Where("cid = ?", id).Find(&cateArticleList).Count(&total).Error
	if err != nil {
		return nil, errmsg.ERROR_CATEGORY_NOTEXIST, 0
	}
	return cateArticleList, errmsg.SUCCESS, total
}

// GetArticleInfo 查询单个文章
func GetArticleInfo(id int) (Article, int) {
	var article Article
	err := db.Preload("Category").Where("id = ?", id).First(&article).Error
	if err != nil {
		return article, errmsg.ERROR_ARTICLE_NOTFOUND
	}
	return article, errmsg.SUCCESS
}

func GetArticle(pageSize int, pageNum int) ([]Article, int, int64) {
	var articleList []Article
	var total int64
	//预加载preload
	err := db.Preload("Category").Limit(pageSize).Offset((pageNum - 1) * pageSize).Find(&articleList).Count(&total).Error
	if err != nil {
		return nil, errmsg.ERROR, 0
	}
	return articleList, errmsg.SUCCESS, total
}

func EditArticle(id int, data *Article) int {
	var maps = make(map[string]interface{})
	maps["title"] = data.Title
	maps["Cid"] = data.Cid
	maps["Desc"] = data.Desc
	maps["Content"] = data.Content
	maps["Img"] = data.Img
	err := db.Model(&Article{}).Where("id = ?", id).Updates(maps).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}

func DeleteArticle(id int) int {
	var article Article
	err := db.Where("id = ?", id).Delete(&article).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}
