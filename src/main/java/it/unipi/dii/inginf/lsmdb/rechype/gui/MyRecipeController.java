package it.unipi.dii.inginf.lsmdb.rechype.gui;

import it.unipi.dii.inginf.lsmdb.rechype.JSONAdder;
import it.unipi.dii.inginf.lsmdb.rechype.drink.Drink;
import it.unipi.dii.inginf.lsmdb.rechype.profile.Profile;
import it.unipi.dii.inginf.lsmdb.rechype.profile.ProfileService;
import it.unipi.dii.inginf.lsmdb.rechype.profile.ProfileServiceFactory;
import it.unipi.dii.inginf.lsmdb.rechype.recipe.Recipe;
import it.unipi.dii.inginf.lsmdb.rechype.user.UserService;
import it.unipi.dii.inginf.lsmdb.rechype.user.UserServiceFactory;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Orientation;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Separator;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import org.bson.Document;
import org.json.JSONArray;
import org.json.JSONObject;

import javax.print.Doc;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class MyRecipeController extends JSONAdder implements Initializable {

    @FXML private VBox vboxFood;
    @FXML private VBox vboxDrinks;

    private UserServiceFactory userServiceFactory;
    private UserService userService;

    private GuiElementsBuilder builder;

    @Override
    public void initialize(URL location, ResourceBundle resources) {

        userServiceFactory = UserServiceFactory.create();
        userService = userServiceFactory.getService();

        builder = new GuiElementsBuilder();

        vboxFood.setSpacing(20);
        vboxDrinks.setSpacing(20);

        Document docUser = userService.getRecipeAndDrinks(userService.getLoggedUser().getUsername());

        List<Document> recipesDoc = (List<Document>) docUser.get("recipes");
        for(Document doc: recipesDoc){
            Recipe rec = new Recipe(doc);
            HBox boxRecipe = builder.createRecipeBlock(rec);
            boxRecipe.setOnMouseClicked((MouseEvent e) ->{
                JSONObject par = new JSONObject().put("_id", rec.getId());
                Main.changeScene("RecipePage", par);
            });
            vboxFood.getChildren().addAll(boxRecipe, new Separator(Orientation.HORIZONTAL));
        }

        List<Document> drinksDoc = (List<Document>) docUser.get("drinks");
        for(Document doc: drinksDoc){
            Drink drink = new Drink(doc);
            HBox boxRecipe = builder.createDrinkBlock(drink);
            boxRecipe.setOnMouseClicked((MouseEvent e) ->{
                JSONObject par = new JSONObject().put("_id", drink.getId());
                Main.changeScene("DrinkPage", par);
            });
            vboxDrinks.getChildren().addAll(boxRecipe, new Separator(Orientation.HORIZONTAL));
        }


    }

    @Override
    public void setGui(){
        JSONObject par = jsonParameters;
        if(par.toString().equals("{}"))
            return;
    }

}
