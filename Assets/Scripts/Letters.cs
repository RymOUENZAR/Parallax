using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Letters : MonoBehaviour
{
    public Transform[] povs;
    public GameObject[] letters;
    public float range;
    public GameObject winPanel;
    private int win = 0;

    void Start()
    {
        for (int i = 0; i < letters.Length; ++i)
        {
            letters[i].SetActive(false);
        }
    }
    

    void Update()
    {
        // Check distance to key points of view
        for (int i = 0; i < povs.Length; ++i)
        {
            if ((povs[i].position - transform.position).magnitude < range)
            {
                if (!letters[i].activeSelf)
                    win++;

                letters[i].SetActive(true);
            }
        }

        if(win == 3)
        {
            win += 1;
            winPanel.SetActive(true);
        }
    }
}
