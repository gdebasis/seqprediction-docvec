/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package indexer;

/**
 *
 * @author Debasis
 */

public interface AMI_FIELDS {
    String FIELD_SENTENCE_ID = "id"; // globally unique id (doc_name.segmentid.sentenceid) 
    String FIELD_DOC_NAME = "docname";
    String FIELD_CONTENT = "content";  // analyzed content
    String FIELD_DECISION_SCORE = "decision_score";
    String FIELD_PREF_SCORE = "pref_score";
    String FIELD_SPEAKER_ID = "speakerid";
}

