package model

import (
	"blog/utils/errmsg"
)

type Category struct {
	ID   uint   `gorm:"primary_key;auto_increment" json:"id"`
	Name string `gorm:"type: varchar(20); not null" json:"name"`
}

func CheckCategory(name string) int {
	var category Category
	db.Select("id").Where("name = ?", name).First(&category)

	if category.ID > 0 {
		return errmsg.ERROR_CATEGORY_USED // code:1001
	}
	return errmsg.SUCCESS
}
func CreateCategory(data *Category) int {
	err := db.Create(&data).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}
func GetCategory(pageSize int, pageNum int) ([]Category, int64) {
	var category []Category
	var total int64
	err := db.Model(&Category{}).Limit(pageSize).
		Offset((pageNum - 1) * pageSize).Find(&category).Count(&total).Error
	if err != nil {
		return nil, 0
	}
	return category, total
}
func EditCategory(id int, data *Category) int {
	var maps = make(map[string]interface{})
	maps["name"] = data.Name

	err := db.Model(&Category{}).Where("id = ?", id).Updates(maps).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}
func DeleteCategory(id int) int {
	var category Category
	err := db.Where("id = ?", id).Delete(&category).Error
	if err != nil {
		return errmsg.ERROR
	}
	return errmsg.SUCCESS
}
