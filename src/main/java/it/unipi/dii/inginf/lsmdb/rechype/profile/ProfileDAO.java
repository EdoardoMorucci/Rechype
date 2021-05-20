package it.unipi.dii.inginf.lsmdb.rechype.profile;

import com.mongodb.MongoException;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import it.unipi.dii.inginf.lsmdb.rechype.persistence.MongoDriver;
import it.unipi.dii.inginf.lsmdb.rechype.user.User;
import org.apache.logging.log4j.LogManager;
import org.bson.Document;

import java.util.ArrayList;
import java.util.List;

import static com.mongodb.client.model.Filters.eq;

class ProfileDAO {

    //retrieving a profile's entity specifying the username
    public Profile getProfileByUsername(String username){
        try(MongoCursor<Document> cursor = MongoDriver.getObject().getCollection(MongoDriver.Collections.PROFILES).find(eq("_id", username)).iterator()){
            Document doc = cursor.next();
            Profile retProf = new Profile(doc);
        }catch(MongoException me){
            LogManager.getLogger("UserDao.class").error("MongoDB: an error occurred");
            System.exit(-1);
        }
        return null;
    }

    //it inserts the profile's entity along with its vectors of objects "meals" and "fridge"
    public Boolean insertProfile(String username){

        List<Document> meals = new ArrayList<>();
        List<Document> fridge = new ArrayList<>();

        Document doc = new Document().append("_id", username).append("meals", meals).append("fridge", fridge);
        try {
            MongoCollection<Document> coll=MongoDriver.getObject().getCollection(MongoDriver.Collections.PROFILES);
            coll.insertOne(doc);
            return true;

        } catch (MongoException me) {
            LogManager.getLogger("UserDao.class").error("MongoDB: an error occurred");
        }
        return false;
    }

    //delete a profile's entity
    public Boolean deleteProfile(String username){
        MongoCollection<Document> coll=null;
        try{
            coll=MongoDriver.getObject().getCollection(MongoDriver.Collections.PROFILES);
            coll.deleteOne(eq("_id", username));
            return true;
        }catch(MongoException ex){
            LogManager.getLogger("ProfileDAO.class").error("MongoDB[PARSE], profile deletion failed: "+username);
            return false;
        }
    }

}
