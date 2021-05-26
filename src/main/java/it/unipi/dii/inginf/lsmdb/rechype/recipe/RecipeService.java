package it.unipi.dii.inginf.lsmdb.rechype.recipe;

import it.unipi.dii.inginf.lsmdb.rechype.user.User;
import org.bson.Document;
import org.json.JSONObject;

import java.util.List;

public interface RecipeService {

    String addRecipe(Document doc);
    List<Recipe> searchRecipe(String text, int offset, int quantity, JSONObject filters);
    JSONObject getCachedRecipe(String key);
    void putRecipeInCache(Document recipe);
    String addLike(String _id, String username);
    Document searchRecipeById(String id);
    String removeLike(String _id, String username);

}
